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

