from __future__ import annotations

import numpy as np


def time_order_split(
    n_rows: int,
    *,
    valid_fraction: float = 0.2,
    gap_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 10:
        raise ValueError("样本量太少，无法做时间顺序切分")
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction 必须在 0 和 1 之间")
    if gap_rows < 0:
        raise ValueError("gap_rows 不能为负数")

    valid_size = max(1, int(round(n_rows * valid_fraction)))
    valid_start = n_rows - valid_size
    train_end = valid_start - gap_rows
    if train_end <= 0:
        raise ValueError("训练区间为空，请减小 valid_fraction 或 gap_rows")

    train_idx = np.arange(0, train_end)
    valid_idx = np.arange(valid_start, n_rows)
    return train_idx, valid_idx


def time_series_cv_splits(
    n_rows: int,
    *,
    n_splits: int = 5,
    gap_rows: int = 0,
) -> list[dict[str, object]]:
    if n_splits < 2:
        raise ValueError("n_splits 必须至少为 2")
    if gap_rows < 0:
        raise ValueError("gap_rows 不能为负数")
    if n_rows < n_splits + 1:
        raise ValueError("样本量太少，无法按指定折数做时间 CV")

    boundaries = np.linspace(0, n_rows, n_splits + 2, dtype=int)
    splits: list[dict[str, object]] = []
    for fold in range(1, n_splits + 1):
        train_start = 0
        valid_start = int(boundaries[fold])
        valid_end = int(boundaries[fold + 1])
        train_end = valid_start - gap_rows
        if train_end <= train_start:
            raise ValueError("训练区间为空，请减少 n_splits 或 gap_rows")
        if valid_end <= valid_start:
            raise ValueError("验证区间为空，请减少 n_splits")

        splits.append(
            {
                "fold": fold,
                "train_start": train_start,
                "train_end": train_end - 1,
                "valid_start": valid_start,
                "valid_end": valid_end - 1,
                "train_idx": np.arange(train_start, train_end),
                "valid_idx": np.arange(valid_start, valid_end),
            }
        )

    return splits
