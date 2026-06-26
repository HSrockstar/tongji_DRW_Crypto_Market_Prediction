from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl

from preprocess import DEFAULT_ROOT, ensure_dir, raw_path, save_json


def inspect_parquet(path: Path) -> dict:
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    numeric_columns = [column for column, dtype in schema.items() if dtype.is_numeric()]
    row_count = int(lf.select(pl.len()).collect().item())

    exprs = []
    for column in numeric_columns:
        exprs.append(pl.col(column).is_null().sum().alias(f"{column}__null"))
        exprs.append(pl.col(column).is_nan().sum().alias(f"{column}__nan"))
        exprs.append(pl.col(column).is_infinite().sum().alias(f"{column}__inf"))

    row = lf.select(exprs).collect().row(0, named=True)
    totals = {"null": 0, "nan": 0, "inf": 0}
    affected_columns = []
    for column in numeric_columns:
        null_count = int(row[f"{column}__null"])
        nan_count = int(row[f"{column}__nan"])
        inf_count = int(row[f"{column}__inf"])
        totals["null"] += null_count
        totals["nan"] += nan_count
        totals["inf"] += inf_count
        if null_count or nan_count or inf_count:
            affected_columns.append(
                {
                    "column": column,
                    "null": null_count,
                    "nan": nan_count,
                    "inf": inf_count,
                    "total": null_count + nan_count + inf_count,
                }
            )

    affected_columns.sort(key=lambda item: item["total"], reverse=True)
    return {
        "path": str(path),
        "row_count": row_count,
        "column_count": len(schema),
        "numeric_column_count": len(numeric_columns),
        "total_null": totals["null"],
        "total_nan": totals["nan"],
        "total_inf": totals["inf"],
        "affected_column_count": len(affected_columns),
        "affected_columns_top": affected_columns[:30],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 DRW 数据中的缺失值、NaN 和无穷值")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument(
        "--output",
        default="outputs/experiments/missing_inf_report.json",
        help="检查结果 JSON 输出路径，相对于项目根目录",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    report = {
        "project_root": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": {},
    }

    for file_name in ["train.parquet", "test.parquet"]:
        info = inspect_parquet(raw_path(root, file_name))
        report["files"][file_name] = info
        print(
            f"{file_name}: null={info['total_null']}, "
            f"NaN={info['total_nan']}, inf/-inf={info['total_inf']}, "
            f"异常列数={info['affected_column_count']}"
        )

    output_path = root / args.output
    ensure_dir(output_path.parent)
    save_json(report, output_path)
    print(f"检查报告已保存: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

