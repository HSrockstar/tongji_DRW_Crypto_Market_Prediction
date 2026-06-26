from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq


DEFAULT_ROOT = Path(r"E:\DRW")
TARGET_COL = "label"
INDEX_COL = "__index_level_0__"
DEFAULT_DROP_COLS = {TARGET_COL, INDEX_COL}


def resolve_root(root: str | Path | None = None) -> Path:
    return Path(root or DEFAULT_ROOT).expanduser().resolve()


def raw_path(root: str | Path, file_name: str) -> Path:
    path = resolve_root(root) / "data" / "raw" / file_name
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return path


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_parquet_columns(path: str | Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def get_feature_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column not in DEFAULT_DROP_COLS]


def load_parquet_frame(
    root: str | Path,
    file_name: str,
    *,
    sample_rows: int | None = None,
    include_label: bool = True,
    selected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """按原始行顺序读取 parquet，并尽早转换为 float32 降低内存占用。"""
    path = raw_path(root, file_name)
    columns = get_parquet_columns(path)
    if selected_columns is None:
        selected_columns = columns
    selected_columns = [column for column in selected_columns if column in columns]
    if not include_label and TARGET_COL in selected_columns:
        selected_columns.remove(TARGET_COL)
    if INDEX_COL in selected_columns:
        selected_columns.remove(INDEX_COL)

    scan = pl.scan_parquet(path).select(selected_columns)
    if sample_rows is not None:
        scan = scan.head(sample_rows)

    schema = scan.collect_schema()
    cast_exprs = []
    for column in selected_columns:
        dtype = schema[column]
        if dtype.is_numeric():
            cast_exprs.append(pl.col(column).cast(pl.Float32).alias(column))
        else:
            cast_exprs.append(pl.col(column))

    frame = scan.select(cast_exprs).collect().to_pandas()
    return frame


def validate_no_missing_or_infinite(data: pd.DataFrame, columns: list[str], *, context: str) -> None:
    """当前数据应无缺失和无穷；若后续新特征引入异常，直接暴露问题。"""
    if not columns:
        raise ValueError(f"{context}: 特征列为空")

    values = data[columns].to_numpy(dtype=np.float32, copy=False)
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    if nan_count or inf_count:
        raise ValueError(f"{context}: 检测到 NaN={nan_count}, inf/-inf={inf_count}")


def save_json(data: dict, path: str | Path) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

