from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.second_place_features import (  # noqa: E402
    DEFAULT_ASSET_DIR,
    DEFAULT_CACHE_DIR,
    PROJECT_ROOT,
    TEST_CACHE_NAME,
    TRAIN_CACHE_NAME,
    read_json,
    resolve_path,
    sha256_file,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "second_place"
DEFAULT_SUBMISSION_DIR = PROJECT_ROOT / "outputs" / "submissions"
DEFAULT_REFERENCE_MANIFEST = DEFAULT_ASSET_DIR / "second_place_reference_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证第 2 名方案迁移结果与独立目录参考结果一致")
    parser.add_argument("--reference-manifest", default=str(DEFAULT_REFERENCE_MANIFEST), help="参考结果清单")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="主项目缓存目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="主项目实验输出目录")
    parser.add_argument("--submission-dir", default=str(DEFAULT_SUBMISSION_DIR), help="主项目提交输出目录")
    parser.add_argument("--skip-oof-hash", action="store_true", help="跳过 OOF 文件 hash 对比")
    parser.add_argument("--float-tol", type=float, default=1e-12, help="CV 浮点指标容忍误差")
    return parser.parse_args()


def compare_hashes(section: str, expected: dict[str, str], actual_paths: dict[str, Path]) -> list[str]:
    errors: list[str] = []
    for key, expected_hash in expected.items():
        path = actual_paths[key]
        if not path.is_file():
            errors.append(f"{section}.{key}: 文件不存在: {path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            errors.append(f"{section}.{key}: SHA256 不一致，actual={actual_hash}, expected={expected_hash}")
    return errors


def values_equal(actual: Any, expected: Any, tol: float) -> bool:
    if expected is None:
        return pd.isna(actual)
    if isinstance(expected, float):
        if math.isnan(expected):
            return pd.isna(actual)
        return abs(float(actual) - expected) <= tol
    if isinstance(expected, int):
        return int(actual) == expected
    return str(actual) == str(expected)


def compare_table(path: Path, expected_rows: list[dict[str, Any]], ignored_columns: set[str], tol: float) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"表格不存在: {path}"]
    actual_df = pd.read_csv(path)
    if len(actual_df) != len(expected_rows):
        return [f"{path.name}: 行数不一致，actual={len(actual_df)}, expected={len(expected_rows)}"]
    for row_index, expected in enumerate(expected_rows):
        actual = actual_df.iloc[row_index].to_dict()
        for column, expected_value in expected.items():
            if column in ignored_columns:
                continue
            if column not in actual:
                errors.append(f"{path.name} row {row_index}: 缺少列 {column}")
                continue
            if not values_equal(actual[column], expected_value, tol):
                errors.append(
                    f"{path.name} row {row_index} column {column}: "
                    f"actual={actual[column]!r}, expected={expected_value!r}"
                )
    return errors


def main() -> int:
    args = parse_args()
    reference_manifest = resolve_path(args.reference_manifest, DEFAULT_REFERENCE_MANIFEST)
    cache_dir = resolve_path(args.cache_dir, DEFAULT_CACHE_DIR)
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    submission_dir = resolve_path(args.submission_dir, DEFAULT_SUBMISSION_DIR)
    manifest = read_json(reference_manifest)

    errors: list[str] = []
    hashes = manifest["hashes"]
    errors.extend(
        compare_hashes(
            "cache",
            hashes["cache"],
            {
                TRAIN_CACHE_NAME: cache_dir / TRAIN_CACHE_NAME,
                TEST_CACHE_NAME: cache_dir / TEST_CACHE_NAME,
            },
        )
    )
    errors.extend(
        compare_hashes(
            "submissions",
            hashes["submissions"],
            {
                "linear": submission_dir / "submission_second_place_linear.csv",
                "ridge": submission_dir / "submission_second_place_ridge.csv",
                "lightgbm": submission_dir / "submission_second_place_lightgbm.csv",
            },
        )
    )
    if not args.skip_oof_hash and "oof" in hashes:
        errors.extend(
            compare_hashes(
                "oof",
                hashes["oof"],
                {
                    "ridge": output_dir / "oof" / "oof_predictions_ridge.csv",
                    "lightgbm": output_dir / "oof" / "oof_predictions_lightgbm.csv",
                },
            )
        )

    if "cv_results" in manifest:
        errors.extend(
            compare_table(
                output_dir / "cv_results.csv",
                manifest["cv_results"],
                {"elapsed_seconds"},
                args.float_tol,
            )
        )
    if "cv_summary" in manifest:
        errors.extend(compare_table(output_dir / "cv_summary.csv", manifest["cv_summary"], set(), args.float_tol))

    if errors:
        print("迁移验收失败:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("迁移验收通过：缓存、提交和 CV 指标均与参考结果一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
