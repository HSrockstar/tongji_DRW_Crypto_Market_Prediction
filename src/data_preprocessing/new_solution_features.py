from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_ASSET_DIR = PROJECT_ROOT / "data" / "external"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "new_solution"
FEATURE_SPEC_NAME = "new_solution_feature_spec.json"
TIME_FILTER_NAME = "new_solution_time_filter.csv"
TRAIN_CACHE_NAME = "train_filtered_450.parquet"
TEST_CACHE_NAME = "test_450.parquet"
FEATURE_LIST_NAME = "feature_list.json"
MANIFEST_NAME = "data_manifest.json"

EXPECTED_TRAIN_ROWS = 525_886
EXPECTED_TEST_ROWS = 538_150
EXPECTED_FILTERED_ROWS = 71_282
EXPECTED_FEATURE_COUNT = 450
EXPECTED_NEGATIVE_FEATURE_COUNT = 150
EXPECTED_POSITIVE_FEATURE_COUNT = 150
AGGREGATION_WINDOWS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]

Logger = Callable[[str], None]


TIME_FOLDS: list[dict[str, Any]] = [
    {
        "fold": 1,
        "train_start": "2023-03-01",
        "valid_start": "2023-07-01",
        "valid_end": "2023-09-01",
        "label": "train_2023-03_to_2023-06_valid_2023-07_to_2023-08",
    },
    {
        "fold": 2,
        "train_start": "2023-03-01",
        "valid_start": "2023-09-01",
        "valid_end": "2023-11-01",
        "label": "train_2023-03_to_2023-08_valid_2023-09_to_2023-10",
    },
    {
        "fold": 3,
        "train_start": "2023-03-01",
        "valid_start": "2023-11-01",
        "valid_end": "2024-01-01",
        "label": "train_2023-03_to_2023-10_valid_2023-11_to_2023-12",
    },
    {
        "fold": 4,
        "train_start": "2023-03-01",
        "valid_start": "2024-01-01",
        "valid_end": "2024-03-01",
        "label": "train_2023-03_to_2023-12_valid_2024-01_to_2024-02",
    },
]


class FileLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(line + "\n")


def resolve_path(path_text: str | None, default: Path) -> Path:
    path = Path(path_text) if path_text else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_feature_spec(asset_dir: Path) -> dict[str, Any]:
    spec_path = asset_dir / FEATURE_SPEC_NAME
    spec = read_json(spec_path)
    negative_features = spec["negative_features"]
    positive_features = spec["positive_features"]
    final_features = spec["final_features"]
    if len(negative_features) != EXPECTED_NEGATIVE_FEATURE_COUNT:
        raise ValueError(f"negative 特征数量不符合预期: {len(negative_features)}")
    if len(positive_features) != EXPECTED_POSITIVE_FEATURE_COUNT:
        raise ValueError(f"positive 特征数量不符合预期: {len(positive_features)}")
    if len(final_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"最终特征数量不符合预期: {len(final_features)}")
    if len(set(final_features)) != EXPECTED_FEATURE_COUNT:
        raise ValueError("最终特征列表存在重复项")
    return spec


def read_parquet_frame(path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    if sample_rows is None:
        return pd.read_parquet(path)
    if sample_rows <= 0:
        raise ValueError("sample rows 必须为正整数")
    parquet_file = pq.ParquetFile(path)
    batch_iter = parquet_file.iter_batches(batch_size=sample_rows)
    try:
        batch = next(batch_iter)
    except StopIteration as exc:
        raise ValueError(f"Parquet 文件为空: {path}") from exc
    return batch.to_pandas()


def reduce_memory_usage(df: pd.DataFrame, logger: Logger | None = None, name: str = "data") -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**3
    if logger:
        logger(f"{name} 初始内存: {start_mem:.2f} GB")

    for col in df.select_dtypes(include=["float"]).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
            df[col] = df[col].astype(np.float16)
        elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**3
    if logger:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem else 0.0
        logger(f"{name} 内存降至: {end_mem:.2f} GB，降低 {reduction:.1f}%")
    return df


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    bid_qty = df["bid_qty"]
    ask_qty = df["ask_qty"]
    buy_qty = df["buy_qty"]
    sell_qty = df["sell_qty"]
    volume = df["volume"]
    eps = 1e-10

    df["bid_ask_interaction"] = bid_qty * ask_qty
    df["bid_buy_interaction"] = bid_qty * buy_qty
    df["bid_sell_interaction"] = bid_qty * sell_qty
    df["ask_buy_interaction"] = ask_qty * buy_qty
    df["ask_sell_interaction"] = ask_qty * sell_qty

    df["volume_weighted_sell"] = sell_qty * volume
    df["buy_sell_ratio"] = buy_qty / (sell_qty + eps)
    df["selling_pressure"] = sell_qty / (volume + eps)
    df["log_volume"] = np.log1p(volume)

    df["effective_spread_proxy"] = np.abs(buy_qty - sell_qty) / (volume + eps)
    df["bid_ask_imbalance"] = (bid_qty - ask_qty) / (bid_qty + ask_qty + eps)
    df["order_flow_imbalance"] = (buy_qty - sell_qty) / (buy_qty + sell_qty + eps)
    df["liquidity_ratio"] = (bid_qty + ask_qty) / (volume + eps)

    df["net_order_flow"] = buy_qty - sell_qty
    df["normalized_net_flow"] = df["net_order_flow"] / (volume + eps)
    df["buying_pressure"] = buy_qty / (volume + eps)
    df["volume_weighted_buy"] = buy_qty * volume

    df["total_depth"] = bid_qty + ask_qty
    df["depth_imbalance"] = (bid_qty - ask_qty) / (df["total_depth"] + eps)
    df["relative_spread"] = np.abs(bid_qty - ask_qty) / (df["total_depth"] + eps)
    df["log_depth"] = np.log1p(df["total_depth"])

    df["kyle_lambda"] = np.abs(df["net_order_flow"]) / (volume + eps)
    df["flow_toxicity"] = np.abs(df["order_flow_imbalance"]) * volume
    df["aggressive_flow_ratio"] = (buy_qty + sell_qty) / (df["total_depth"] + eps)

    df["volume_depth_ratio"] = volume / (df["total_depth"] + eps)
    df["activity_intensity"] = (buy_qty + sell_qty) / (volume + eps)
    df["log_buy_qty"] = np.log1p(buy_qty)
    df["log_sell_qty"] = np.log1p(sell_qty)
    df["log_bid_qty"] = np.log1p(bid_qty)
    df["log_ask_qty"] = np.log1p(ask_qty)

    df["realized_spread_proxy"] = 2 * np.abs(df["net_order_flow"]) / (volume + eps)
    df["price_impact_proxy"] = df["net_order_flow"] / (df["total_depth"] + eps)
    df["quote_volatility_proxy"] = np.abs(df["depth_imbalance"])

    df["flow_depth_interaction"] = df["net_order_flow"] * df["total_depth"]
    df["imbalance_volume_interaction"] = df["order_flow_imbalance"] * volume
    df["depth_volume_interaction"] = df["total_depth"] * volume
    df["buy_sell_spread"] = np.abs(buy_qty - sell_qty)
    df["bid_ask_spread"] = np.abs(bid_qty - ask_qty)

    df["trade_informativeness"] = df["net_order_flow"] / (bid_qty + ask_qty + eps)
    df["execution_shortfall_proxy"] = df["buy_sell_spread"] / (volume + eps)
    df["adverse_selection_proxy"] = df["net_order_flow"] / (df["total_depth"] + eps) * volume

    df["fill_probability"] = volume / (buy_qty + sell_qty + eps)
    df["execution_rate"] = (buy_qty + sell_qty) / (df["total_depth"] + eps)
    df["market_efficiency"] = volume / (df["bid_ask_spread"] + eps)

    df["sqrt_volume"] = np.sqrt(volume)
    df["sqrt_depth"] = np.sqrt(df["total_depth"])
    df["volume_squared"] = volume**2
    df["imbalance_squared"] = df["order_flow_imbalance"] ** 2

    df["bid_ratio"] = bid_qty / (df["total_depth"] + eps)
    df["ask_ratio"] = ask_qty / (df["total_depth"] + eps)
    df["buy_ratio"] = buy_qty / (buy_qty + sell_qty + eps)
    df["sell_ratio"] = sell_qty / (buy_qty + sell_qty + eps)

    df["liquidity_consumption"] = (buy_qty + sell_qty) / (df["total_depth"] + eps)
    df["market_stress"] = volume / (df["total_depth"] + eps) * np.abs(df["order_flow_imbalance"])
    df["depth_depletion"] = volume / (bid_qty + ask_qty + eps)

    df["net_buying_ratio"] = df["net_order_flow"] / (volume + eps)
    df["directional_volume"] = df["net_order_flow"] * np.log1p(volume)
    df["signed_volume"] = np.sign(df["net_order_flow"]) * volume
    return df.replace([np.inf, -np.inf], 0).fillna(0)


def create_aggregated_features(df: pd.DataFrame, feature_list: list[str], prefix: str) -> pd.DataFrame:
    df[f"{prefix}_sum"] = df[feature_list].sum(axis=1)
    df[f"{prefix}_mean"] = df[feature_list].mean(axis=1)
    df[f"{prefix}_median"] = df[feature_list].median(axis=1)
    df[f"{prefix}_max"] = df[feature_list].max(axis=1)
    df[f"{prefix}_min"] = df[feature_list].min(axis=1)
    df[f"{prefix}_std"] = df[feature_list].std(axis=1)
    return df


def add_aggregations(
    df: pd.DataFrame,
    negative_features: list[str],
    positive_features: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    for window in windows or AGGREGATION_WINDOWS:
        df = create_aggregated_features(df, negative_features[:window], f"Negative_features_{window}")
        df = create_aggregated_features(df, positive_features[:window], f"Positive_features_{window}")
    return df


def add_new_solution_features(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    df = add_microstructure_features(df)
    return add_aggregations(
        df,
        spec["negative_features"],
        spec["positive_features"],
        spec.get("aggregation_windows", AGGREGATION_WINDOWS),
    )


def normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    if "index" in df.columns:
        return df.rename(columns={"index": "timestamp"})
    if "__index_level_0__" in df.columns:
        return df.rename(columns={"__index_level_0__": "timestamp"})
    first_col = df.columns[0]
    return df.rename(columns={first_col: "timestamp"})


def add_time_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["Y_M_D_H"] = df["timestamp"].dt.strftime("%Y-%m-%d-%H")
    df["Y_M_D_H_M"] = df["timestamp"].dt.strftime("%Y-%m-%d-%H-%M")
    return df


def load_time_filter(asset_dir: Path) -> pd.Series:
    time_filter = pd.read_csv(asset_dir / TIME_FILTER_NAME)
    parsed_times = pd.to_datetime(time_filter.values.ravel(), errors="coerce")
    if parsed_times.isna().any():
        raise ValueError("时间过滤 CSV 中存在无法解析的时间戳")
    return parsed_times.strftime("%Y-%m-%d-%H-%M")


def filter_training_rows_by_time(train_df: pd.DataFrame, asset_dir: Path) -> pd.DataFrame:
    train_df = normalize_timestamp_column(train_df)
    train_df = add_time_filter_columns(train_df)
    selected_times = load_time_filter(asset_dir)
    return train_df.loc[train_df["Y_M_D_H_M"].isin(selected_times)]


def ensure_features_exist(df: pd.DataFrame, features: list[str], context: str) -> None:
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        preview = ", ".join(missing[:20])
        raise ValueError(f"{context} 缺少 {len(missing)} 个最终特征，前 20 个: {preview}")


def finite_frame_values(df: pd.DataFrame, features: list[str], context: str) -> None:
    values = df[features].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{context} 的最终特征矩阵包含 NaN 或 Inf")


def cast_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[features] = out[features].astype(np.float32)
    return out


def parquet_shape(path: Path) -> tuple[int, int]:
    parquet_file = pq.ParquetFile(path)
    return parquet_file.metadata.num_rows, parquet_file.metadata.num_columns
