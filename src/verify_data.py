import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"E:\DRW")
RAW_FILES = ["train.parquet", "test.parquet", "sample_submission.csv"]
PARQUET_FILES = ["train.parquet", "test.parquet"]
MARKET_FIELDS = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1024 / 1024, 2)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_parquet_metadata(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(f"无法导入 pyarrow.parquet: {type(exc).__name__}: {exc}") from exc

    try:
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        columns = parquet_file.schema_arrow.names
    except Exception as exc:
        raise RuntimeError(f"读取 Parquet 元信息失败: {path}，原因: {type(exc).__name__}: {exc}") from exc

    market_found = [field for field in MARKET_FIELDS if field in columns]
    market_missing = [field for field in MARKET_FIELDS if field not in columns]
    return {
        "path": str(path),
        "size_mb": size_mb(path),
        "row_count": metadata.num_rows,
        "column_count": len(columns),
        "columns": columns,
        "first_20_columns": columns[:20],
        "has_label": "label" in columns,
        "market_fields": {
            "checked": MARKET_FIELDS,
            "found": market_found,
            "missing": market_missing,
        },
    }


def read_submission(path: Path) -> tuple[dict[str, Any], str]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(f"无法导入 pandas: {type(exc).__name__}: {exc}") from exc

    try:
        data = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"读取 CSV 失败: {path}，原因: {type(exc).__name__}: {exc}") from exc

    head_text = data.head().to_string(index=False)
    info = {
        "path": str(path),
        "size_mb": size_mb(path),
        "row_count": int(data.shape[0]),
        "column_count": int(data.shape[1]),
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "columns": list(data.columns),
        "head": data.head().to_dict(orient="records"),
    }
    return info, head_text


def print_parquet_info(name: str, info: dict[str, Any]) -> None:
    print(f"{name}:")
    print(f"  文件路径: {info['path']}")
    print(f"  文件大小 MB: {info['size_mb']}")
    print(f"  行数: {info['row_count']}")
    print(f"  列数: {info['column_count']}")
    print(f"  列名: {json.dumps(info['columns'], ensure_ascii=False)}")
    print(f"  前 20 个列名: {json.dumps(info['first_20_columns'], ensure_ascii=False)}")
    print(f"  是否包含 label 列: {info['has_label']}")
    print(f"  常见市场字段存在: {json.dumps(info['market_fields']['found'], ensure_ascii=False)}")
    print(f"  常见市场字段缺失: {json.dumps(info['market_fields']['missing'], ensure_ascii=False)}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRW 数据文件校验脚本")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录，默认 E:\\DRW")
    parser.add_argument("--hash", action="store_true", help="计算数据文件 SHA256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    raw_dir = root / "data" / "raw"
    output_dir = root / "outputs"
    report_path = output_dir / "data_check_report.json"

    print(f"项目根目录: {root}")
    print(f"原始数据目录: {raw_dir}")
    print()

    missing_files = [file_name for file_name in RAW_FILES if not (raw_dir / file_name).is_file()]
    if missing_files:
        print("数据文件不存在:")
        for file_name in missing_files:
            print(f"  - {raw_dir / file_name}")
        return 1

    file_info = {}
    print("文件大小:")
    for file_name in RAW_FILES:
        path = raw_dir / file_name
        file_info[file_name] = {
            "path": str(path),
            "size_mb": size_mb(path),
        }
        print(f"  {file_name}: {file_info[file_name]['size_mb']} MB")
    print()

    parquet_info = {}
    for file_name in PARQUET_FILES:
        parquet_info[file_name] = read_parquet_metadata(raw_dir / file_name)
        print_parquet_info(file_name, parquet_info[file_name])

    submission_info, submission_head = read_submission(raw_dir / "sample_submission.csv")
    print("sample_submission.csv:")
    print(f"  文件路径: {submission_info['path']}")
    print(f"  文件大小 MB: {submission_info['size_mb']}")
    print(f"  shape: {tuple(submission_info['shape'])}")
    print(f"  columns: {json.dumps(submission_info['columns'], ensure_ascii=False)}")
    print("  head:")
    print(submission_head)
    print()

    key_checks = {
        "all_required_files_exist": True,
        "train_has_label": parquet_info["train.parquet"]["has_label"],
        "test_has_label": parquet_info["test.parquet"]["has_label"],
        "train_market_fields_found": parquet_info["train.parquet"]["market_fields"]["found"],
        "train_market_fields_missing": parquet_info["train.parquet"]["market_fields"]["missing"],
        "test_market_fields_found": parquet_info["test.parquet"]["market_fields"]["found"],
        "test_market_fields_missing": parquet_info["test.parquet"]["market_fields"]["missing"],
    }

    report = {
        "project_root": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": file_info,
        "train": parquet_info["train.parquet"],
        "test": parquet_info["test.parquet"],
        "sample_submission": submission_info,
        "key_checks": key_checks,
    }

    if args.hash:
        print("开始计算 SHA256:")
        hashes = {}
        for file_name in RAW_FILES:
            path = raw_dir / file_name
            hashes[file_name] = sha256_file(path)
            print(f"  {file_name}: {hashes[file_name]}")
        report["sha256"] = hashes
        print()

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"写入 JSON 校验报告失败: {report_path}，原因: {type(exc).__name__}: {exc}")
        return 1

    print(f"JSON 校验报告已保存: {report_path}")
    print("数据校验完成")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"错误: {exc}")
        raise SystemExit(1)
