from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_FIELDS = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
DERIVED_FEATURES = [
    "book_total_qty",
    "trade_total_qty",
    "book_imbalance",
    "trade_imbalance",
    "volume_per_book_qty",
    "volume_per_trade_qty",
    "log_volume",
]


def add_basic_market_features(data: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """增加只依赖当前行的基础市场派生特征。"""
    missing = [field for field in MARKET_FIELDS if field not in data.columns]
    if missing:
        raise ValueError(f"缺少公开市场字段: {missing}")

    result = data.copy()
    bid_qty = result["bid_qty"].astype("float32")
    ask_qty = result["ask_qty"].astype("float32")
    buy_qty = result["buy_qty"].astype("float32")
    sell_qty = result["sell_qty"].astype("float32")
    volume = result["volume"].astype("float32")

    book_total = bid_qty + ask_qty
    trade_total = buy_qty + sell_qty

    result["book_total_qty"] = book_total
    result["trade_total_qty"] = trade_total
    result["book_imbalance"] = (bid_qty - ask_qty) / (book_total + eps)
    result["trade_imbalance"] = (buy_qty - sell_qty) / (trade_total + eps)
    result["volume_per_book_qty"] = volume / (book_total + eps)
    result["volume_per_trade_qty"] = volume / (trade_total + eps)
    result["log_volume"] = np.log1p(np.maximum(volume.to_numpy(dtype="float32"), 0.0))

    for column in DERIVED_FEATURES:
        result[column] = result[column].astype("float32")

    return result


def add_synthesized_features(
    data: pd.DataFrame,
    combo_defs: list[dict[str, str]],
    *,
    eps: float = 1e-6,
) -> pd.DataFrame:
    result = data.copy()
    for combo in combo_defs:
        name = combo["name"]
        left = combo["left"]
        right = combo["right"]
        op = combo["op"]
        a = result[left].astype("float32")
        b = result[right].astype("float32")
        if op == "add":
            result[name] = a + b
        elif op == "sub":
            result[name] = a - b
        elif op == "div":
            result[name] = a / (np.abs(b) + eps)
        elif op == "mul":
            result[name] = a * b
        else:
            raise ValueError(f"未知组合操作: {op}")
        result[name] = result[name].astype("float32")
    return result

