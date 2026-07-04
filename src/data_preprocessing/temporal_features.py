from __future__ import annotations

import numpy as np
import pandas as pd

from data_preprocessing.build_features import add_basic_market_features

# Step4 原版 Top-5
TOP_X_LAG_BASIC = ["X466", "X33", "X752", "X272", "X758"]
# Kaggle 对齐版：importance Top-10
TOP_X_LAG_EXTENDED = [
    "X466",
    "X33",
    "X752",
    "X272",
    "X758",
    "X738",
    "X280",
    "X611",
    "X614",
    "X778",
]
MARKET_TEMPORAL_BASIC = ["volume", "book_imbalance", "trade_imbalance"]
MARKET_TEMPORAL_EXTENDED = ["volume", "book_imbalance", "trade_imbalance", "log_volume"]


def _add_market_temporal(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    include_diff: bool,
) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column].astype(np.float32)
        for lag in lags:
            frame[f"{column}_lag{lag}"] = series.shift(lag).fillna(0.0).astype(np.float32)
        for window in rolling_windows:
            frame[f"{column}_rollmean{window}"] = (
                series.rolling(window, min_periods=1).mean().astype(np.float32)
            )
            frame[f"{column}_rollstd{window}"] = (
                series.rolling(window, min_periods=1).std().fillna(0.0).astype(np.float32)
            )
        if include_diff:
            frame[f"{column}_diff1"] = series.diff().fillna(0.0).astype(np.float32)
    return frame


def _add_x_lags(frame: pd.DataFrame, features: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    for feature in features:
        if feature not in frame.columns:
            continue
        series = frame[feature].astype(np.float32)
        for lag in lags:
            frame[f"{feature}_lag{lag}"] = series.shift(lag).fillna(0.0).astype(np.float32)
    return frame


def add_step4_temporal_features(data: pd.DataFrame, *, extended: bool = False) -> pd.DataFrame:
    """Step4 时序特征；extended=True 为 Kaggle 对齐增强版。"""
    result = add_basic_market_features(data)
    if extended:
        result = _add_market_temporal(
            result,
            MARKET_TEMPORAL_EXTENDED,
            lags=(1, 3, 5, 10),
            rolling_windows=(5, 10),
            include_diff=True,
        )
        result = _add_x_lags(result, TOP_X_LAG_EXTENDED, lags=(1, 5))
    else:
        result = _add_market_temporal(
            result,
            MARKET_TEMPORAL_BASIC,
            lags=(1, 5, 10),
            rolling_windows=(5,),
            include_diff=False,
        )
        result = _add_x_lags(result, TOP_X_LAG_BASIC, lags=(1,))
    return result
