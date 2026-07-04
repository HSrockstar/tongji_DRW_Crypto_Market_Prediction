from __future__ import annotations

import numpy as np


def default_purge_gap_rows(n_rows: int, *, n_groups: int = 6) -> int:
    """近似冠军方案 gap=1 组（约 2 个月）的行数隔离。"""
    if n_rows < n_groups:
        return 0
    return max(1, n_rows // n_groups)


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


def purged_group_time_series_splits(
    n_rows: int,
    *,
    n_groups: int = 6,
    gap_groups: int = 1,
) -> list[dict[str, object]]:
    """Purged expanding-window CV：按时间均分为 n_groups 组，验证组前留 gap_groups 组隔离。

    与冠军描述的「6 groups + gap=1」一致：训练集只使用验证组之前、且与验证组间隔 gap 的数据。
    """
    if n_groups < 3:
        raise ValueError("n_groups 必须至少为 3")
    if gap_groups < 0:
        raise ValueError("gap_groups 不能为负数")
    if n_rows < n_groups + gap_groups + 1:
        raise ValueError("样本量太少，无法做 purged group CV")

    boundaries = np.linspace(0, n_rows, n_groups + 1, dtype=int)
    splits: list[dict[str, object]] = []
    fold = 0
    for valid_group in range(1 + gap_groups, n_groups):
        fold += 1
        train_start = 0
        train_end = int(boundaries[valid_group - gap_groups]) - 1
        valid_start = int(boundaries[valid_group])
        valid_end = int(boundaries[valid_group + 1]) - 1
        if train_end < train_start:
            continue
        if valid_end < valid_start:
            continue

        splits.append(
            {
                "fold": fold,
                "valid_group": valid_group,
                "train_start": train_start,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "train_idx": np.arange(train_start, train_end + 1),
                "valid_idx": np.arange(valid_start, valid_end + 1),
                "gap_groups": gap_groups,
                "n_groups": n_groups,
            }
        )

    if len(splits) < 2:
        raise ValueError("有效折数不足，请减小 gap_groups 或增大样本量")
    return splits
