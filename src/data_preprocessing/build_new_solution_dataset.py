from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.new_solution_features import (  # noqa: E402
    DEFAULT_ASSET_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_RAW_DATA_DIR,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_FILTERED_ROWS,
    EXPECTED_TEST_ROWS,
    FEATURE_LIST_NAME,
    MANIFEST_NAME,
    TEST_CACHE_NAME,
    TRAIN_CACHE_NAME,
    FileLogger,
    add_new_solution_features,
    cast_feature_frame,
    ensure_features_exist,
    filter_training_rows_by_time,
    finite_frame_values,
    load_feature_spec,
    parquet_shape,
    read_parquet_frame,
    reduce_memory_usage,
    resolve_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建新方案迁移版 450 特征缓存")
    parser.add_argument("--raw-data-dir", default=str(DEFAULT_RAW_DATA_DIR), help="原始数据目录")
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR), help="迁移版资产目录")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="缓存输出目录")
    parser.add_argument("--sample-train-rows", type=int, default=None, help="仅读取前 N 行 train 做 smoke 检查")
    parser.add_argument("--sample-test-rows", type=int, default=None, help="仅读取前 N 行 test 做 smoke 检查")
    parser.add_argument("--test-batch-size", type=int, default=25_000, help="完整测试集分块行数")
    parser.add_argument("--force", action="store_true", help="即使缓存存在也重新生成")
    return parser.parse_args()


def cache_is_valid(cache_dir: Path) -> bool:
    train_path = cache_dir / TRAIN_CACHE_NAME
    test_path = cache_dir / TEST_CACHE_NAME
    feature_path = cache_dir / FEATURE_LIST_NAME
    if not (train_path.is_file() and test_path.is_file() and feature_path.is_file()):
        return False
    try:
        train_rows, train_cols = parquet_shape(train_path)
        test_rows, test_cols = parquet_shape(test_path)
    except Exception:
        return False
    return (
        train_rows == EXPECTED_FILTERED_ROWS
        and train_cols == EXPECTED_FEATURE_COUNT + 2
        and test_rows == EXPECTED_TEST_ROWS
        and test_cols == EXPECTED_FEATURE_COUNT
    )


def build_train_cache(
    raw_data_dir: Path,
    asset_dir: Path,
    cache_dir: Path,
    spec: dict[str, object],
    logger: FileLogger,
    sample_rows: int | None,
) -> Path:
    features = list(spec["final_features"])  # type: ignore[index]
    train_path = raw_data_dir / "train.parquet"
    logger.write("读取 train.parquet")
    train_df = read_parquet_frame(train_path, sample_rows)
    train_df = reduce_memory_usage(train_df, logger.write, "train")

    logger.write("执行新方案迁移版公开市场特征工程")
    train_df = add_new_solution_features(train_df, spec)

    logger.write("按迁移版时间过滤 CSV 筛选训练样本")
    train_clean = filter_training_rows_by_time(train_df, asset_dir)
    if train_clean.empty:
        raise ValueError("时间过滤后训练集为空；smoke 检查请增大 --sample-train-rows")
    if sample_rows is None and len(train_clean) != EXPECTED_FILTERED_ROWS:
        raise ValueError(f"筛选训练样本数不符合预期: {len(train_clean)}")

    ensure_features_exist(train_clean, features, "训练缓存")
    finite_frame_values(train_clean, features, "训练缓存")
    train_cache = train_clean[["timestamp", "label", *features]].copy()
    train_cache = cast_feature_frame(train_cache, features)
    train_cache["label"] = train_cache["label"].astype(np.float32)

    train_cache_path = cache_dir / TRAIN_CACHE_NAME
    logger.write(f"写出训练缓存: {train_cache_path}")
    train_cache.to_parquet(train_cache_path, index=False, compression="snappy")
    del train_df, train_clean, train_cache
    gc.collect()
    return train_cache_path


def build_test_cache_full(
    raw_data_dir: Path,
    cache_dir: Path,
    spec: dict[str, object],
    logger: FileLogger,
    batch_size: int,
) -> Path:
    features = list(spec["final_features"])  # type: ignore[index]
    test_path = raw_data_dir / "test.parquet"
    test_cache_path = cache_dir / TEST_CACHE_NAME
    if test_cache_path.exists():
        test_cache_path.unlink()

    logger.write("分块处理 test.parquet")
    writer: pq.ParquetWriter | None = None
    test_rows = 0
    parquet_file = pq.ParquetFile(test_path)
    for batch_index, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size), start=1):
        logger.write(f"处理 test batch {batch_index}: rows={batch.num_rows}")
        test_df = batch.to_pandas()
        test_df = reduce_memory_usage(test_df, logger.write, f"test_batch_{batch_index}")
        test_df = add_new_solution_features(test_df, spec)
        ensure_features_exist(test_df, features, f"测试缓存 batch {batch_index}")
        finite_frame_values(test_df, features, f"测试缓存 batch {batch_index}")
        test_cache = test_df[features].copy()
        test_cache = cast_feature_frame(test_cache, features)
        table = pa.Table.from_pandas(test_cache, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(test_cache_path, table.schema, compression="snappy")
        writer.write_table(table)
        test_rows += len(test_cache)
        del test_df, test_cache, table
        gc.collect()
    if writer is not None:
        writer.close()
    if test_rows != EXPECTED_TEST_ROWS:
        raise ValueError(f"测试缓存行数不符合预期: {test_rows}")
    return test_cache_path


def build_test_cache_sample(
    raw_data_dir: Path,
    cache_dir: Path,
    spec: dict[str, object],
    logger: FileLogger,
    sample_rows: int,
) -> Path:
    features = list(spec["final_features"])  # type: ignore[index]
    test_path = raw_data_dir / "test.parquet"
    logger.write("读取 test.parquet smoke 样本")
    test_df = read_parquet_frame(test_path, sample_rows)
    test_df = reduce_memory_usage(test_df, logger.write, "test")
    test_df = add_new_solution_features(test_df, spec)
    ensure_features_exist(test_df, features, "测试缓存")
    finite_frame_values(test_df, features, "测试缓存")
    test_cache = cast_feature_frame(test_df[features].copy(), features)
    test_cache_path = cache_dir / TEST_CACHE_NAME
    logger.write(f"写出测试缓存: {test_cache_path}")
    test_cache.to_parquet(test_cache_path, index=False, compression="snappy")
    del test_df, test_cache
    gc.collect()
    return test_cache_path


def write_cache_metadata(
    raw_data_dir: Path,
    asset_dir: Path,
    cache_dir: Path,
    spec: dict[str, object],
    train_cache_path: Path,
    test_cache_path: Path,
    elapsed_seconds: float,
    sample_train_rows: int | None,
    sample_test_rows: int | None,
) -> None:
    features = list(spec["final_features"])  # type: ignore[index]
    feature_info = {
        "feature_count": len(features),
        "features": features,
        "negative_features_count": len(spec["negative_features"]),  # type: ignore[arg-type]
        "positive_features_count": len(spec["positive_features"]),  # type: ignore[arg-type]
    }
    write_json(feature_info, cache_dir / FEATURE_LIST_NAME)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_data_dir": raw_data_dir,
        "asset_dir": asset_dir,
        "train_filtered_rows": parquet_shape(train_cache_path)[0],
        "test_rows": parquet_shape(test_cache_path)[0],
        "feature_count": len(features),
        "train_cache": train_cache_path,
        "test_cache": test_cache_path,
        "train_cache_sha256": sha256_file(train_cache_path),
        "test_cache_sha256": sha256_file(test_cache_path),
        "sample_train_rows": sample_train_rows,
        "sample_test_rows": sample_test_rows,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    write_json(manifest, cache_dir / MANIFEST_NAME)


def main() -> int:
    args = parse_args()
    raw_data_dir = resolve_path(args.raw_data_dir, DEFAULT_RAW_DATA_DIR)
    asset_dir = resolve_path(args.asset_dir, DEFAULT_ASSET_DIR)
    cache_dir = resolve_path(args.cache_dir, DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = FileLogger(cache_dir / "prepare_log.txt")
    start = time.perf_counter()

    logger.write(f"原始数据目录: {raw_data_dir}")
    logger.write(f"资产目录: {asset_dir}")
    logger.write(f"缓存目录: {cache_dir}")

    if (
        not args.force
        and args.sample_train_rows is None
        and args.sample_test_rows is None
        and cache_is_valid(cache_dir)
    ):
        logger.write("检测到有效完整缓存，跳过重建；如需重建请加 --force")
        return 0

    spec = load_feature_spec(asset_dir)
    train_cache_path = build_train_cache(raw_data_dir, asset_dir, cache_dir, spec, logger, args.sample_train_rows)
    if args.sample_test_rows is None:
        test_cache_path = build_test_cache_full(raw_data_dir, cache_dir, spec, logger, args.test_batch_size)
    else:
        test_cache_path = build_test_cache_sample(raw_data_dir, cache_dir, spec, logger, args.sample_test_rows)

    elapsed = time.perf_counter() - start
    write_cache_metadata(
        raw_data_dir,
        asset_dir,
        cache_dir,
        spec,
        train_cache_path,
        test_cache_path,
        elapsed,
        args.sample_train_rows,
        args.sample_test_rows,
    )
    logger.write(f"缓存构建完成，总耗时 {elapsed:.1f} 秒")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
