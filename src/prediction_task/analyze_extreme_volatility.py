from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import DERIVED_FEATURES, MARKET_FIELDS, add_basic_market_features
from data_preprocessing.preprocess import (  # noqa: E402
    DEFAULT_ROOT,
    TARGET_COL,
    ensure_dir,
    get_parquet_columns,
    load_json,
    load_parquet_frame,
    validate_no_missing_or_infinite,
)
from prediction_task.metrics import evaluate_regression  # noqa: E402
from prediction_task.splits import time_order_split  # noqa: E402
from prediction_task.train_lgbm import save_lgbm_model  # noqa: E402


GROUP_OVERALL = "整体"
GROUP_NORMAL = "普通波动"
GROUP_MEDIUM = "中等波动"
GROUP_EXTREME = "异常波动"
MODEL_LABELS = {
    "ridge": "Ridge",
    "lightgbm": "LightGBM",
    "weighted_lightgbm": "Weighted LightGBM",
    "continuous_weighted_lightgbm": "连续权重 Weighted LightGBM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="异常波动样本加权建模与误差分析")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--gap-rows", type=int, default=0, help="训练集和验证集之间的 gap 行数")
    parser.add_argument("--num-boost-round", type=int, default=3000, help="Weighted LightGBM 最大迭代轮数")
    parser.add_argument("--early-stopping-rounds", type=int, default=200, help="早停轮数")
    parser.add_argument("--log-period", type=int, default=50, help="LightGBM 日志间隔")
    parser.add_argument("--normal-weight", type=float, default=1.0, help="普通波动训练样本权重")
    parser.add_argument("--medium-weight", type=float, default=1.5, help="中等波动训练样本权重")
    parser.add_argument("--extreme-weight", type=float, default=3.0, help="异常波动训练样本权重")
    parser.add_argument(
        "--weight-mode",
        choices=["bucket", "continuous"],
        default="bucket",
        help="样本权重模式：bucket 为三档权重，continuous 为连续权重",
    )
    parser.add_argument("--continuous-tail-quantile", type=float, default=0.99, help="连续权重达到封顶值的训练集分位数")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，直接读取已有 Weighted LightGBM 预测")
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def lgb_pearson_eval(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    labels = dataset.get_label()
    metrics = evaluate_regression(labels, preds)
    return "pearson", metrics["pearson"], True


def load_training_frame(root: Path, feature_cols: list[str]) -> pd.DataFrame:
    parquet_cols = set(get_parquet_columns(root / "data" / "raw" / "train.parquet"))
    raw_feature_cols = [column for column in feature_cols if column not in DERIVED_FEATURES and column in parquet_cols]
    selected_columns = sorted(set(raw_feature_cols) | set(MARKET_FIELDS) | {TARGET_COL})
    data = load_parquet_frame(root, "train.parquet", include_label=True, selected_columns=selected_columns)
    data = add_basic_market_features(data)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="异常波动 Weighted LightGBM 训练数据")
    return data


def build_train_weights(
    y_train: np.ndarray,
    *,
    normal_weight: float,
    medium_weight: float,
    extreme_weight: float,
    weight_mode: str = "bucket",
    continuous_tail_quantile: float = 0.99,
) -> tuple[np.ndarray, dict[str, float | int]]:
    abs_label = np.abs(y_train)
    q80 = float(np.quantile(abs_label, 0.80))
    q95 = float(np.quantile(abs_label, 0.95))
    medium_mask = (abs_label > q80) & (abs_label <= q95)
    extreme_mask = abs_label > q95
    if weight_mode == "bucket":
        weights = np.full(len(y_train), normal_weight, dtype=np.float32)
        weights[medium_mask] = medium_weight
        weights[extreme_mask] = extreme_weight
        tail_quantile = q95
    elif weight_mode == "continuous":
        if continuous_tail_quantile <= 0.95 or continuous_tail_quantile >= 1.0:
            raise ValueError("--continuous-tail-quantile 必须位于 (0.95, 1.0) 区间内")
        tail_quantile = float(np.quantile(abs_label, continuous_tail_quantile))
        first_span = max(q95 - q80, np.finfo(float).eps)
        tail_span = max(tail_quantile - q95, np.finfo(float).eps)
        weights = np.full(len(y_train), normal_weight, dtype=np.float64)
        medium_progress = np.clip((abs_label - q80) / first_span, 0.0, 1.0)
        tail_progress = np.clip((abs_label - q95) / tail_span, 0.0, 1.0)
        weights += medium_progress * (medium_weight - normal_weight)
        weights += tail_progress * (extreme_weight - medium_weight)
        weights = np.clip(weights, min(normal_weight, medium_weight, extreme_weight), max(normal_weight, medium_weight, extreme_weight))
        weights = weights.astype(np.float32)
    else:
        raise ValueError(f"未知权重模式: {weight_mode}")
    info = {
        "weight_mode": weight_mode,
        "train_abs_label_q80": q80,
        "train_abs_label_q95": q95,
        "continuous_tail_quantile": continuous_tail_quantile,
        "train_abs_label_tail_threshold": tail_quantile,
        "normal_weight": normal_weight,
        "medium_weight": medium_weight,
        "extreme_weight": extreme_weight,
        "normal_train_rows": int((abs_label <= q80).sum()),
        "medium_train_rows": int(medium_mask.sum()),
        "extreme_train_rows": int(extreme_mask.sum()),
        "weight_min": float(weights.min()),
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "weight_max": float(weights.max()),
    }
    return weights, info


def model_key_for_weight_mode(weight_mode: str) -> str:
    if weight_mode == "bucket":
        return "weighted_lightgbm"
    if weight_mode == "continuous":
        return "continuous_weighted_lightgbm"
    raise ValueError(f"未知权重模式: {weight_mode}")


def file_stem_for_weight_mode(weight_mode: str) -> str:
    if weight_mode == "bucket":
        return "extreme_volatility_weighted_lgbm"
    if weight_mode == "continuous":
        return "extreme_volatility_continuous_weighted_lgbm"
    raise ValueError(f"未知权重模式: {weight_mode}")


def train_weighted_lgbm(
    root: Path,
    output_dir: Path,
    model_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_cols = load_json(model_dir / "official_lgbm_features.json")["feature_columns"]
    data = load_training_frame(root, feature_cols)
    train_idx, valid_idx = time_order_split(len(data), valid_fraction=args.valid_fraction, gap_rows=args.gap_rows)
    x_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    x_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)

    sample_weight, weight_info = build_train_weights(
        y_train,
        normal_weight=args.normal_weight,
        medium_weight=args.medium_weight,
        extreme_weight=args.extreme_weight,
        weight_mode=args.weight_mode,
        continuous_tail_quantile=args.continuous_tail_quantile,
    )
    params = load_json(output_dir / "official_lgbm_params.json")["params"]
    params = dict(params)
    params["metric"] = "None"

    train_set = lgb.Dataset(x_train, label=y_train, weight=sample_weight, feature_name=feature_cols)
    valid_set = lgb.Dataset(x_valid, label=y_valid, feature_name=feature_cols, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        feval=lgb_pearson_eval,
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(args.log_period),
        ],
    )
    best_iteration = int(model.best_iteration or args.num_boost_round)
    valid_pred = model.predict(x_valid, num_iteration=best_iteration)
    prediction = pd.DataFrame(
        {
            "row_index": valid_idx,
            "y_true": y_valid,
            "y_pred": valid_pred,
            "model": model_key_for_weight_mode(args.weight_mode),
        }
    )
    file_stem = file_stem_for_weight_mode(args.weight_mode)
    pred_path = output_dir / f"{file_stem}_valid_predictions.csv"
    prediction.to_csv(pred_path, index=False)
    model_path = model_dir / f"{file_stem}.txt"
    save_lgbm_model(model, model_path)

    training_info: dict[str, object] = {
        **weight_info,
        "model_key": model_key_for_weight_mode(args.weight_mode),
        "weighted_lgbm_best_iteration": best_iteration,
        "weighted_lgbm_prediction_csv": str(pred_path),
        "weighted_lgbm_model_path": str(model_path),
        "weighted_lgbm_overall_metrics": evaluate_regression(y_valid, valid_pred),
    }
    return prediction, training_info


def validate_prediction_alignment(frames: dict[str, pd.DataFrame]) -> np.ndarray:
    reference_name = next(iter(frames))
    reference = frames[reference_name]
    y_true = reference["y_true"].to_numpy(dtype=np.float64)
    for model_name, frame in frames.items():
        if len(frame) != len(reference):
            raise ValueError(f"{model_name} 验证预测行数不一致")
        if not frame["row_index"].equals(reference["row_index"]):
            raise ValueError(f"{model_name} 验证预测 row_index 不一致")
        if not np.allclose(frame["y_true"].to_numpy(dtype=np.float64), y_true, rtol=0, atol=1e-12):
            raise ValueError(f"{model_name} 验证预测 y_true 不一致")
    return y_true


def validation_group_masks(y_true: np.ndarray, group_stats: pd.DataFrame) -> tuple[float, float, list[tuple[str, np.ndarray, str]]]:
    q80 = float(group_stats["abs_label_q80_threshold"].iloc[0])
    q95 = float(group_stats["abs_label_q95_threshold"].iloc[0])
    abs_label = np.abs(y_true)
    groups = [
        (GROUP_OVERALL, np.ones(len(y_true), dtype=bool), "all validation samples"),
        (GROUP_NORMAL, abs_label <= q80, f"abs(label) <= {q80:.12g}"),
        (GROUP_MEDIUM, (abs_label > q80) & (abs_label <= q95), f"{q80:.12g} < abs(label) <= {q95:.12g}"),
        (GROUP_EXTREME, abs_label > q95, f"abs(label) > {q95:.12g}"),
    ]
    return q80, q95, groups


def compute_metrics_and_contribution(
    frames: dict[str, pd.DataFrame],
    group_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = validate_prediction_alignment(frames)
    _, _, groups = validation_group_masks(y_true, group_stats)
    metric_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []

    for model_name, frame in frames.items():
        y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
        residual = y_true - y_pred
        squared_error = residual**2
        absolute_error = np.abs(residual)
        total_squared_error = float(squared_error.sum())
        total_absolute_error = float(absolute_error.sum())
        for group_name, mask, definition in groups:
            metrics = evaluate_regression(y_true[mask], y_pred[mask])
            squared_error_sum = float(squared_error[mask].sum())
            absolute_error_sum = float(absolute_error[mask].sum())
            row = {
                "model": model_name,
                "model_label": MODEL_LABELS.get(model_name, model_name),
                "group": group_name,
                "definition": definition,
                "sample_count": int(mask.sum()),
                "sample_ratio": float(mask.mean()),
                "pearson": metrics["pearson"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "squared_error_sum": squared_error_sum,
                "absolute_error_sum": absolute_error_sum,
                "squared_error_contribution": squared_error_sum / total_squared_error,
                "absolute_error_contribution": absolute_error_sum / total_absolute_error,
            }
            metric_rows.append(row)
            if group_name != GROUP_OVERALL:
                contribution_rows.append(row.copy())

    return pd.DataFrame(metric_rows), pd.DataFrame(contribution_rows)


def save_figures(
    metrics: pd.DataFrame,
    contribution: pd.DataFrame,
    y_true: np.ndarray,
    figure_dir: Path,
) -> list[str]:
    ensure_dir(figure_dir)
    setup_style()
    saved: list[str] = []
    group_order = [GROUP_NORMAL, GROUP_MEDIUM, GROUP_EXTREME]
    preferred_model_order = ["Ridge", "LightGBM", "Weighted LightGBM", "连续权重 Weighted LightGBM"]
    available_labels = set(metrics["model_label"])
    model_order = [label for label in preferred_model_order if label in available_labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(np.abs(y_true), bins=80, ax=ax, color="#4C72B0")
    ax.axvline(float(metrics.loc[metrics["group"] == GROUP_NORMAL, "definition"].iloc[0].split("<=")[-1]), color="#DD8452", linestyle="--", label="80% 分位")
    q95_text = metrics.loc[metrics["group"] == GROUP_MEDIUM, "definition"].iloc[0].split("<=")[-1]
    ax.axvline(float(q95_text), color="#C44E52", linestyle="--", label="95% 分位")
    ax.set_yscale("log")
    ax.set_title("验证集 abs(label) 分布")
    ax.set_xlabel("abs(label)")
    ax.set_ylabel("样本数（log）")
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "label_abs_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))

    grouped = metrics[metrics["group"].isin(group_order)].copy()
    grouped["model_label"] = pd.Categorical(grouped["model_label"], categories=model_order, ordered=True)
    grouped["group"] = pd.Categorical(grouped["group"], categories=group_order, ordered=True)
    for metric_name, file_name, title, ylabel in [
        ("rmse", "group_rmse_compare.png", "不同模型分组 RMSE 对比", "RMSE"),
        ("mae", "group_mae_compare.png", "不同模型分组 MAE 对比", "MAE"),
        ("pearson", "group_pearson_compare.png", "不同模型分组 Pearson 对比", "Pearson"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=grouped, x="group", y=metric_name, hue="model_label", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("波动分组")
        ax.set_ylabel(ylabel)
        ax.legend(title="模型")
        fig.tight_layout()
        path = figure_dir / file_name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    contribution_plot = contribution.copy()
    contribution_plot["model_label"] = pd.Categorical(contribution_plot["model_label"], categories=model_order, ordered=True)
    contribution_plot["group"] = pd.Categorical(contribution_plot["group"], categories=group_order, ordered=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=contribution_plot, x="group", y="squared_error_contribution", hue="model_label", ax=ax)
    ax.set_title("不同波动区间平方误差贡献")
    ax.set_xlabel("波动分组")
    ax.set_ylabel("平方误差贡献占比")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="模型")
    fig.tight_layout()
    path = figure_dir / "error_contribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))

    target_model = "continuous_weighted_lightgbm" if "continuous_weighted_lightgbm" in set(metrics["model"]) else "weighted_lightgbm"
    delta_rows = build_weighted_delta_rows(metrics, target_model=target_model)
    delta = pd.DataFrame(delta_rows)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = []
    for metric, value in zip(delta["metric"], delta["delta"], strict=True):
        lower_is_better = "RMSE" in metric or "MAE" in metric
        improved = value < 0 if lower_is_better else value > 0
        colors.append("#55A868" if improved else "#DD8452")
    ax.bar(delta["metric"], delta["delta"], color=colors, edgecolor="#333333")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title(f"{MODEL_LABELS.get(target_model, target_model)} 相对普通 LightGBM 的指标变化")
    ax.set_ylabel(f"{MODEL_LABELS.get(target_model, target_model)} - LightGBM")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    path = figure_dir / "weighted_lgbm_delta.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))
    return saved


def build_weighted_delta_rows(metrics: pd.DataFrame, target_model: str = "weighted_lightgbm") -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    pairs = [
        (GROUP_OVERALL, "整体"),
        (GROUP_NORMAL, "普通"),
        (GROUP_MEDIUM, "中等"),
        (GROUP_EXTREME, "异常"),
    ]
    for group_name, short_name in pairs:
        base = metrics[(metrics["model"] == "lightgbm") & (metrics["group"] == group_name)].iloc[0]
        weighted = metrics[(metrics["model"] == target_model) & (metrics["group"] == group_name)].iloc[0]
        rows.extend(
            [
                {"metric": f"{short_name} Pearson", "delta": float(weighted["pearson"] - base["pearson"])},
                {"metric": f"{short_name} RMSE", "delta": float(weighted["rmse"] - base["rmse"])},
                {"metric": f"{short_name} MAE", "delta": float(weighted["mae"] - base["mae"])},
            ]
        )
    return rows


def build_weight_scheme_delta_rows(metrics: pd.DataFrame) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    if not {"weighted_lightgbm", "continuous_weighted_lightgbm"}.issubset(set(metrics["model"])):
        return rows
    for group_name in [GROUP_OVERALL, GROUP_NORMAL, GROUP_MEDIUM, GROUP_EXTREME]:
        bucket = metrics[(metrics["model"] == "weighted_lightgbm") & (metrics["group"] == group_name)].iloc[0]
        continuous = metrics[(metrics["model"] == "continuous_weighted_lightgbm") & (metrics["group"] == group_name)].iloc[0]
        rows.extend(
            [
                {"group": group_name, "metric": "pearson", "delta": float(continuous["pearson"] - bucket["pearson"])},
                {"group": group_name, "metric": "rmse", "delta": float(continuous["rmse"] - bucket["rmse"])},
                {"group": group_name, "metric": "mae", "delta": float(continuous["mae"] - bucket["mae"])},
                {
                    "group": group_name,
                    "metric": "squared_error_contribution",
                    "delta": float(continuous["squared_error_contribution"] - bucket["squared_error_contribution"]),
                },
                {
                    "group": group_name,
                    "metric": "absolute_error_contribution",
                    "delta": float(continuous["absolute_error_contribution"] - bucket["absolute_error_contribution"]),
                },
            ]
        )
    return rows


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in formats:
                values.append(formats[column].format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    metrics: pd.DataFrame,
    contribution: pd.DataFrame,
    training_info: dict[str, object],
    figure_paths: list[str],
    target_model: str = "weighted_lightgbm",
) -> None:
    ensure_dir(report_path.parent)
    target_label = MODEL_LABELS.get(target_model, target_model)
    overall = metrics[metrics["group"] == GROUP_OVERALL][["model_label", "pearson", "rmse", "mae"]].copy()
    extreme = metrics[metrics["group"] == GROUP_EXTREME][
        ["model_label", "pearson", "rmse", "mae", "squared_error_contribution", "absolute_error_contribution"]
    ].copy()
    lgbm_extreme = extreme[extreme["model_label"] == "LightGBM"].iloc[0]
    weighted_extreme = extreme[extreme["model_label"] == target_label].iloc[0]
    lgbm_overall = overall[overall["model_label"] == "LightGBM"].iloc[0]
    weighted_overall = overall[overall["model_label"] == target_label].iloc[0]
    rmse_delta = float(weighted_extreme["rmse"] - lgbm_extreme["rmse"])
    mae_delta = float(weighted_extreme["mae"] - lgbm_extreme["mae"])
    pearson_delta = float(weighted_overall["pearson"] - lgbm_overall["pearson"])
    if rmse_delta < 0 or mae_delta < 0:
        conclusion = "Weighted LightGBM 在异常波动组的部分误差指标出现下降，说明样本加权对异常样本有一定改善作用。"
    else:
        conclusion = "Weighted LightGBM 未能降低异常波动组 RMSE/MAE，说明仅通过样本加权不足以改善该组预测误差。"

    overall_table = markdown_table(
        overall,
        ["model_label", "pearson", "rmse", "mae"],
        {"pearson": "{:.6f}", "rmse": "{:.6f}", "mae": "{:.6f}"},
    )
    extreme_table = markdown_table(
        extreme,
        ["model_label", "pearson", "rmse", "mae", "squared_error_contribution", "absolute_error_contribution"],
        {
            "pearson": "{:.6f}",
            "rmse": "{:.6f}",
            "mae": "{:.6f}",
            "squared_error_contribution": "{:.2%}",
            "absolute_error_contribution": "{:.2%}",
        },
    )
    contribution_table = markdown_table(
        contribution[contribution["model"].isin(["lightgbm", target_model])][
            ["model_label", "group", "sample_ratio", "squared_error_contribution", "absolute_error_contribution", "rmse", "mae"]
        ],
        ["model_label", "group", "sample_ratio", "squared_error_contribution", "absolute_error_contribution", "rmse", "mae"],
        {
            "sample_ratio": "{:.2%}",
            "squared_error_contribution": "{:.2%}",
            "absolute_error_contribution": "{:.2%}",
            "rmse": "{:.6f}",
            "mae": "{:.6f}",
        },
    )
    figure_lines = "\n".join(f"- `{path}`" for path in figure_paths)
    content = f"""# 异常波动样本加权建模实验报告

生成时间：{datetime.now().isoformat(timespec="seconds")}

## 实验设置

- 验证划分：按时间顺序前 80% 训练、后 20% 验证。
- 验证分组：普通波动为 `abs(label)` 前 80%，中等波动为 80%-95%，异常波动为后 5%。
- 加权训练阈值仅由训练集 `abs(label)` 分位数计算。
- 目标模型：`{target_label}`。
- 权重模式：`{training_info.get("weight_mode", "bucket")}`。
- 权重锚点：普通 `{training_info["normal_weight"]}`，中等 `{training_info["medium_weight"]}`，异常 `{training_info["extreme_weight"]}`。
- 训练集权重阈值：80% 分位 `{training_info["train_abs_label_q80"]:.6f}`，95% 分位 `{training_info["train_abs_label_q95"]:.6f}`。
- Weighted LightGBM 最佳迭代轮数：`{training_info["weighted_lgbm_best_iteration"]}`。

## 整体验证指标

{overall_table}

## 异常波动组指标

{extreme_table}

## LightGBM 与 Weighted LightGBM 误差贡献

{contribution_table}

## 结果分析

- 相比普通 LightGBM，{target_label} 的整体 Pearson 变化为 `{pearson_delta:+.6f}`。
- 异常波动组 RMSE 变化为 `{rmse_delta:+.6f}`，MAE 变化为 `{mae_delta:+.6f}`。
- {conclusion}
- 从误差贡献看，异常波动样本占验证集约 5%，但在树模型中贡献了明显更高比例的平方误差，说明异常行情是主要误差来源之一。

## 图表文件

{figure_lines}
"""
    report_path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    model_dir = ensure_dir(root / "models")
    figure_dir = ensure_dir(root / "outputs" / "figures" / "extreme_volatility")
    report_dir = ensure_dir(root / "outputs" / "reports")

    target_model = model_key_for_weight_mode(args.weight_mode)
    file_stem = file_stem_for_weight_mode(args.weight_mode)
    weighted_pred_path = output_dir / f"{file_stem}_valid_predictions.csv"
    summary_path = output_dir / f"{file_stem}_summary.json"
    if args.skip_train:
        if not weighted_pred_path.is_file():
            raise FileNotFoundError(f"未找到已有 Weighted LightGBM 预测: {weighted_pred_path}")
        weighted_pred = pd.read_csv(weighted_pred_path)
        if summary_path.is_file():
            training_info = load_json(summary_path).get("training_info", {})
        else:
            training_info = {}
        training_info["weighted_lgbm_prediction_csv"] = str(weighted_pred_path)
    else:
        weighted_pred, training_info = train_weighted_lgbm(root, output_dir, model_dir, args)

    frames = {
        "ridge": pd.read_csv(output_dir / "official_baseline_valid_predictions.csv"),
        "lightgbm": pd.read_csv(output_dir / "official_lgbm_valid_predictions.csv"),
    }
    bucket_pred_path = output_dir / "extreme_volatility_weighted_lgbm_valid_predictions.csv"
    if args.weight_mode == "continuous" and bucket_pred_path.is_file():
        frames["weighted_lightgbm"] = pd.read_csv(bucket_pred_path)
    frames[target_model] = weighted_pred
    group_stats = pd.read_csv(output_dir / "extreme_volatility_group_stats.csv")
    y_true = validate_prediction_alignment(frames)
    metrics, contribution = compute_metrics_and_contribution(frames, group_stats)

    if args.weight_mode == "bucket":
        metrics_path = output_dir / "extreme_volatility_model_metrics.csv"
        contribution_path = output_dir / "extreme_volatility_error_contribution.csv"
        report_path = report_dir / "extreme_volatility_weighted_lgbm_report.md"
    else:
        metrics_path = output_dir / "extreme_volatility_continuous_weighted_lgbm_model_metrics.csv"
        contribution_path = output_dir / "extreme_volatility_continuous_weighted_lgbm_error_contribution.csv"
        report_path = report_dir / "extreme_volatility_continuous_weighted_lgbm_report.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    contribution.to_csv(contribution_path, index=False, encoding="utf-8-sig")
    weight_scheme_delta_rows = build_weight_scheme_delta_rows(metrics)
    weight_scheme_delta_path = output_dir / "extreme_volatility_weight_scheme_delta.csv"
    if weight_scheme_delta_rows:
        pd.DataFrame(weight_scheme_delta_rows).to_csv(weight_scheme_delta_path, index=False, encoding="utf-8-sig")
    figure_paths = save_figures(metrics, contribution, y_true, figure_dir)

    write_report(report_path, metrics, contribution, training_info, figure_paths, target_model=target_model)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics_csv": str(metrics_path),
        "error_contribution_csv": str(contribution_path),
        "weighted_lgbm_prediction_csv": str(weighted_pred_path),
        "report": str(report_path),
        "figures": figure_paths,
        "training_info": training_info,
        "weighted_vs_lgbm_delta": build_weighted_delta_rows(metrics, target_model=target_model),
        "weight_scheme_delta": weight_scheme_delta_rows,
    }
    if weight_scheme_delta_rows:
        summary["weight_scheme_delta_csv"] = str(weight_scheme_delta_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
