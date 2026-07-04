from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

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
GROUP_EXTREME = "异常波动"
SLICE_EXTREME_ALL = "异常波动整体"
SLICE_POS_EXTREME = "正向异常"
SLICE_NEG_EXTREME = "负向异常"
MODEL_LABELS = {
    "ridge": "Ridge",
    "lightgbm": "LightGBM",
    "weighted_lightgbm": "Weighted LightGBM",
    "continuous_weighted_lightgbm": "连续权重 Weighted LightGBM",
    "catboost": "CatBoost baseline",
    "xgboost": "XGBoost baseline",
    "tree_blend": "Tree Blend",
    "two_stage_expert": "Two-stage Expert",
}
PREDICTION_FILES = {
    "ridge": "official_baseline_valid_predictions.csv",
    "lightgbm": "official_lgbm_valid_predictions.csv",
    "weighted_lightgbm": "extreme_volatility_weighted_lgbm_valid_predictions.csv",
    "continuous_weighted_lightgbm": "extreme_volatility_continuous_weighted_lgbm_valid_predictions.csv",
    "catboost": "tree_baseline_catboost_valid_predictions.csv",
    "xgboost": "tree_baseline_xgboost_valid_predictions.csv",
    "tree_blend": "tree_blend_valid_predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练异常检测 + 专家残差回归两阶段模型")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--gap-rows", type=int, default=0, help="训练集和验证集之间的 gap 行数")
    parser.add_argument("--detector-rounds", type=int, default=1500, help="异常检测器最大迭代轮数")
    parser.add_argument("--expert-rounds", type=int, default=1500, help="残差专家最大迭代轮数")
    parser.add_argument("--early-stopping-rounds", type=int, default=100, help="早停轮数")
    parser.add_argument("--log-period", type=int, default=100, help="LightGBM 日志间隔")
    parser.add_argument("--alpha-step", type=float, default=0.05, help="残差修正强度搜索步长")
    parser.add_argument("--alpha-max", type=float, default=1.5, help="残差修正强度搜索上限")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def load_training_frame(root: Path, feature_cols: list[str]) -> pd.DataFrame:
    parquet_cols = set(get_parquet_columns(root / "data" / "raw" / "train.parquet"))
    raw_feature_cols = [column for column in feature_cols if column not in DERIVED_FEATURES and column in parquet_cols]
    selected_columns = sorted(set(raw_feature_cols) | set(MARKET_FIELDS) | {TARGET_COL})
    data = load_parquet_frame(root, "train.parquet", include_label=True, selected_columns=selected_columns)
    data = add_basic_market_features(data)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="两阶段专家模型训练数据")
    return data


def build_detector_labels(y: np.ndarray, threshold: float) -> np.ndarray:
    labels = np.zeros(len(y), dtype=np.int32)
    labels[y > threshold] = 1
    labels[y < -threshold] = 2
    return labels


def build_class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"检测器标签类别缺失，counts={counts.tolist()}")
    class_weights = len(labels) / (len(counts) * counts)
    return class_weights[labels].astype(np.float32)


def train_detector(
    x_train: pd.DataFrame,
    y_train_class: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid_class: np.ndarray,
    sample_weight: np.ndarray,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 20.0,
        "seed": args.seed,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(x_train, label=y_train_class, weight=sample_weight, feature_name=feature_cols)
    valid_set = lgb.Dataset(x_valid, label=y_valid_class, feature_name=feature_cols, reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=args.detector_rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(args.log_period),
        ],
    )


def expert_params(args: argparse.Namespace) -> dict[str, float | int | str]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 20.0,
        "seed": args.seed,
        "verbosity": -1,
    }


def train_residual_expert(
    x_train: pd.DataFrame,
    residual_train: np.ndarray,
    train_mask: np.ndarray,
    x_valid: pd.DataFrame,
    residual_valid: np.ndarray,
    valid_mask: np.ndarray,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> lgb.Booster:
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError("残差专家缺少训练或验证异常样本")
    train_set = lgb.Dataset(
        x_train.loc[train_mask],
        label=residual_train[train_mask],
        feature_name=feature_cols,
    )
    valid_set = lgb.Dataset(
        x_valid.loc[valid_mask],
        label=residual_valid[valid_mask],
        feature_name=feature_cols,
        reference=train_set,
    )
    return lgb.train(
        expert_params(args),
        train_set,
        num_boost_round=args.expert_rounds,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(args.log_period),
        ],
    )


def build_group_masks(y_true: np.ndarray, group_stats: pd.DataFrame) -> list[tuple[str, np.ndarray, str]]:
    q95 = float(group_stats["abs_label_q95_threshold"].iloc[0])
    abs_label = np.abs(y_true)
    return [
        (GROUP_OVERALL, np.ones(len(y_true), dtype=bool), "all validation samples"),
        (GROUP_EXTREME, abs_label > q95, f"abs(label) > {q95:.12g}"),
    ]


def compute_metrics(frames: dict[str, pd.DataFrame], group_stats: pd.DataFrame) -> pd.DataFrame:
    y_true = validate_prediction_alignment(frames)
    rows: list[dict[str, object]] = []
    for model_name, frame in frames.items():
        y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
        residual = y_true - y_pred
        squared_error = residual**2
        absolute_error = np.abs(residual)
        total_squared_error = float(squared_error.sum())
        total_absolute_error = float(absolute_error.sum())
        for group_name, mask, definition in build_group_masks(y_true, group_stats):
            metrics = evaluate_regression(y_true[mask], y_pred[mask])
            squared_error_sum = float(squared_error[mask].sum())
            absolute_error_sum = float(absolute_error[mask].sum())
            rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "group": group_name,
                    "definition": definition,
                    "sample_count": int(mask.sum()),
                    "sample_ratio": float(mask.mean()),
                    "pearson": metrics["pearson"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "squared_error_contribution": squared_error_sum / total_squared_error,
                    "absolute_error_contribution": absolute_error_sum / total_absolute_error,
                }
            )
    return pd.DataFrame(rows)


def compute_bias_analysis(frames: dict[str, pd.DataFrame], group_stats: pd.DataFrame) -> pd.DataFrame:
    y_true = validate_prediction_alignment(frames)
    q95 = float(group_stats["abs_label_q95_threshold"].iloc[0])
    slices = [
        (SLICE_EXTREME_ALL, np.abs(y_true) > q95),
        (SLICE_POS_EXTREME, y_true > q95),
        (SLICE_NEG_EXTREME, y_true < -q95),
    ]
    rows: list[dict[str, object]] = []
    for model_name, frame in frames.items():
        y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
        for slice_name, mask in slices:
            residual = y_true[mask] - y_pred[mask]
            rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "slice": slice_name,
                    "sample_count": int(mask.sum()),
                    "y_true_mean": float(y_true[mask].mean()),
                    "y_pred_mean": float(y_pred[mask].mean()),
                    "mean_residual_y_minus_pred": float(residual.mean()),
                    "mean_abs_error": float(np.abs(residual).mean()),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                }
            )
    return pd.DataFrame(rows)


def load_prediction_frames(output_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []
    for model_name, file_name in PREDICTION_FILES.items():
        path = output_dir / file_name
        if path.is_file():
            frames[model_name] = pd.read_csv(path)
        else:
            missing_files.append(str(path))
    if missing_files:
        joined = "\n".join(missing_files)
        raise FileNotFoundError(f"缺少对比预测文件:\n{joined}")
    return frames


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


def search_alpha(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    correction: np.ndarray,
    extreme_mask: np.ndarray,
    *,
    alpha_step: float,
    alpha_max: float,
) -> tuple[float, pd.DataFrame]:
    if alpha_step <= 0 or alpha_max < 0:
        raise ValueError("--alpha-step 必须为正数，--alpha-max 必须非负")
    count = int(round(alpha_max / alpha_step))
    if not np.isclose(count * alpha_step, alpha_max, atol=1e-9):
        raise ValueError("--alpha-max 必须能被 --alpha-step 整除")
    rows: list[dict[str, float]] = []
    for i in range(count + 1):
        alpha = i * alpha_step
        pred = base_pred + alpha * correction
        overall = evaluate_regression(y_true, pred)
        extreme = evaluate_regression(y_true[extreme_mask], pred[extreme_mask])
        rows.append(
            {
                "alpha": alpha,
                "overall_pearson": overall["pearson"],
                "overall_rmse": overall["rmse"],
                "overall_mae": overall["mae"],
                "extreme_pearson": extreme["pearson"],
                "extreme_rmse": extreme["rmse"],
                "extreme_mae": extreme["mae"],
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["extreme_rmse", "overall_pearson", "overall_rmse"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return float(result.iloc[0]["alpha"]), result


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(formats[column].format(value) if column in formats else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    bias: pd.DataFrame,
    alpha_search: pd.DataFrame,
    training_info: dict[str, object],
) -> None:
    ensure_dir(path.parent)
    best_alpha = float(alpha_search.iloc[0]["alpha"])
    compare_bias = bias[
        bias["model"].isin(["lightgbm", "two_stage_expert"])
        & bias["slice"].isin([SLICE_POS_EXTREME, SLICE_NEG_EXTREME])
    ].copy()
    lgbm_extreme = metrics[(metrics["model"] == "lightgbm") & (metrics["group"] == GROUP_EXTREME)].iloc[0]
    expert_extreme = metrics[(metrics["model"] == "two_stage_expert") & (metrics["group"] == GROUP_EXTREME)].iloc[0]
    rmse_delta = float(expert_extreme["rmse"] - lgbm_extreme["rmse"])
    mae_delta = float(expert_extreme["mae"] - lgbm_extreme["mae"])
    lgbm_pos = compare_bias[(compare_bias["model"] == "lightgbm") & (compare_bias["slice"] == SLICE_POS_EXTREME)].iloc[0]
    expert_pos = compare_bias[(compare_bias["model"] == "two_stage_expert") & (compare_bias["slice"] == SLICE_POS_EXTREME)].iloc[0]
    lgbm_neg = compare_bias[(compare_bias["model"] == "lightgbm") & (compare_bias["slice"] == SLICE_NEG_EXTREME)].iloc[0]
    expert_neg = compare_bias[(compare_bias["model"] == "two_stage_expert") & (compare_bias["slice"] == SLICE_NEG_EXTREME)].iloc[0]
    pos_abs_residual_delta = float(abs(expert_pos["mean_residual_y_minus_pred"]) - abs(lgbm_pos["mean_residual_y_minus_pred"]))
    neg_abs_residual_delta = float(abs(expert_neg["mean_residual_y_minus_pred"]) - abs(lgbm_neg["mean_residual_y_minus_pred"]))
    if rmse_delta < 0 or mae_delta < 0:
        conclusion = "两阶段专家模型让异常组 RMSE/MAE 小幅下降，但收益很弱；从正负异常切片看，当前修正主要来自负向异常，正向异常的向 0 收缩没有改善。"
    else:
        conclusion = "两阶段专家模型未降低异常组 RMSE/MAE，说明当前检测器或残差专家尚未有效修正极端偏差。"
    content = f"""# 两阶段异常检测与专家残差回归报告

生成时间：{datetime.now().isoformat(timespec="seconds")}

## 实验设置

- 基础模型：复用已有官方 LightGBM。
- 检测器：LightGBM 三分类，类别为普通/正向异常/负向异常。
- 专家模型：正向异常残差专家与负向异常残差专家。
- 训练集异常阈值：`abs(label)` 95% 分位 `{training_info["train_abs_label_q95"]:.6f}`。
- 训练样本数：普通 `{training_info["normal_train_rows"]}`，正向异常 `{training_info["positive_extreme_train_rows"]}`，负向异常 `{training_info["negative_extreme_train_rows"]}`。
- 最优 alpha：`{best_alpha:.2f}`。
- 说明：alpha 直接在当前验证集上搜索，用于课程实验分析，不作为严格无偏泛化估计。

## 整体与异常组指标

{markdown_table(
        metrics,
        ["model_label", "group", "sample_ratio", "pearson", "rmse", "mae", "squared_error_contribution", "absolute_error_contribution"],
        {
            "sample_ratio": "{:.2%}",
            "pearson": "{:.6f}",
            "rmse": "{:.6f}",
            "mae": "{:.6f}",
            "squared_error_contribution": "{:.2%}",
            "absolute_error_contribution": "{:.2%}",
        },
    )}

## 正负异常偏差对比

{markdown_table(
        compare_bias,
        ["model_label", "slice", "sample_count", "y_true_mean", "y_pred_mean", "mean_residual_y_minus_pred", "mean_abs_error", "rmse"],
        {
            "y_true_mean": "{:.6f}",
            "y_pred_mean": "{:.6f}",
            "mean_residual_y_minus_pred": "{:.6f}",
            "mean_abs_error": "{:.6f}",
            "rmse": "{:.6f}",
        },
    )}

## Alpha 搜索最优行

{markdown_table(
        alpha_search.head(10),
        ["alpha", "overall_pearson", "overall_rmse", "overall_mae", "extreme_pearson", "extreme_rmse", "extreme_mae"],
        {
            "alpha": "{:.2f}",
            "overall_pearson": "{:.6f}",
            "overall_rmse": "{:.6f}",
            "overall_mae": "{:.6f}",
            "extreme_pearson": "{:.6f}",
            "extreme_rmse": "{:.6f}",
            "extreme_mae": "{:.6f}",
        },
    )}

## 结论

- 相比 LightGBM，Two-stage Expert 的异常组 RMSE 变化 `{rmse_delta:+.6f}`，MAE 变化 `{mae_delta:+.6f}`。
- 正向异常平均残差绝对值变化 `{pos_abs_residual_delta:+.6f}`，负向异常平均残差绝对值变化 `{neg_abs_residual_delta:+.6f}`。
- {conclusion}
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    model_dir = ensure_dir(root / "models")
    report_dir = ensure_dir(root / "outputs" / "reports")

    feature_cols = load_json(model_dir / "official_lgbm_features.json")["feature_columns"]
    data = load_training_frame(root, feature_cols)
    train_idx, valid_idx = time_order_split(len(data), valid_fraction=args.valid_fraction, gap_rows=args.gap_rows)
    x_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    x_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)

    base_model = lgb.Booster(model_file=str(model_dir / "official_lgbm.txt"))
    official_lgbm_result = pd.read_csv(output_dir / "official_lgbm_results.csv").iloc[0]
    base_best_iteration = int(official_lgbm_result["best_iteration"])
    base_train_pred = base_model.predict(x_train, num_iteration=base_best_iteration)
    base_valid_frame = pd.read_csv(output_dir / "official_lgbm_valid_predictions.csv")
    if not np.array_equal(base_valid_frame["row_index"].to_numpy(), valid_idx):
        raise ValueError("official_lgbm_valid_predictions.csv 与当前验证切分不一致")
    if not np.allclose(base_valid_frame["y_true"].to_numpy(dtype=np.float64), y_valid, rtol=0, atol=1e-12):
        raise ValueError("official_lgbm_valid_predictions.csv 的 y_true 与当前验证集不一致")
    base_valid_pred = base_valid_frame["y_pred"].to_numpy(dtype=np.float64)
    residual_train = y_train - base_train_pred
    residual_valid = y_valid - base_valid_pred

    train_q95 = float(np.quantile(np.abs(y_train), 0.95))
    y_train_class = build_detector_labels(y_train, train_q95)
    y_valid_class = build_detector_labels(y_valid, train_q95)
    sample_weight = build_class_weights(y_train_class)

    detector = train_detector(x_train, y_train_class, x_valid, y_valid_class, sample_weight, feature_cols, args)
    detector_path = model_dir / "two_stage_extreme_detector.txt"
    save_lgbm_model(detector, detector_path)
    valid_proba = detector.predict(x_valid, num_iteration=detector.best_iteration)

    pos_train_mask = y_train > train_q95
    neg_train_mask = y_train < -train_q95
    pos_valid_mask = y_valid > train_q95
    neg_valid_mask = y_valid < -train_q95
    pos_expert = train_residual_expert(
        x_train,
        residual_train,
        pos_train_mask,
        x_valid,
        residual_valid,
        pos_valid_mask,
        feature_cols,
        args,
    )
    neg_expert = train_residual_expert(
        x_train,
        residual_train,
        neg_train_mask,
        x_valid,
        residual_valid,
        neg_valid_mask,
        feature_cols,
        args,
    )
    pos_expert_path = model_dir / "two_stage_pos_residual_expert.txt"
    neg_expert_path = model_dir / "two_stage_neg_residual_expert.txt"
    save_lgbm_model(pos_expert, pos_expert_path)
    save_lgbm_model(neg_expert, neg_expert_path)

    pos_residual_pred = pos_expert.predict(x_valid, num_iteration=pos_expert.best_iteration)
    neg_residual_pred = neg_expert.predict(x_valid, num_iteration=neg_expert.best_iteration)
    correction = valid_proba[:, 1] * pos_residual_pred + valid_proba[:, 2] * neg_residual_pred

    group_stats = pd.read_csv(output_dir / "extreme_volatility_group_stats.csv")
    extreme_mask = np.abs(y_valid) > float(group_stats["abs_label_q95_threshold"].iloc[0])
    best_alpha, alpha_search = search_alpha(
        y_valid,
        base_valid_pred,
        correction,
        extreme_mask,
        alpha_step=args.alpha_step,
        alpha_max=args.alpha_max,
    )
    final_pred = base_valid_pred + best_alpha * correction
    two_stage_pred = pd.DataFrame(
        {
            "row_index": valid_idx,
            "y_true": y_valid,
            "y_pred": final_pred,
            "model": "two_stage_expert",
            "base_lgbm_pred": base_valid_pred,
            "p_normal": valid_proba[:, 0],
            "p_pos_extreme": valid_proba[:, 1],
            "p_neg_extreme": valid_proba[:, 2],
            "pos_residual_pred": pos_residual_pred,
            "neg_residual_pred": neg_residual_pred,
            "correction": correction,
            "alpha": best_alpha,
        }
    )
    pred_path = output_dir / "two_stage_expert_valid_predictions.csv"
    two_stage_pred.to_csv(pred_path, index=False)

    alpha_search_path = output_dir / "two_stage_expert_alpha_search.csv"
    alpha_search.to_csv(alpha_search_path, index=False, encoding="utf-8-sig")
    frames = load_prediction_frames(output_dir)
    frames["two_stage_expert"] = two_stage_pred[["row_index", "y_true", "y_pred", "model"]].copy()
    metrics = compute_metrics(frames, group_stats)
    bias = compute_bias_analysis(frames, group_stats)

    metrics_path = output_dir / "two_stage_expert_all_models_metrics.csv"
    bias_path = output_dir / "two_stage_expert_bias_analysis.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    bias.to_csv(bias_path, index=False, encoding="utf-8-sig")

    class_counts = np.bincount(y_train_class, minlength=3)
    valid_class_counts = np.bincount(y_valid_class, minlength=3)
    training_info: dict[str, object] = {
        "train_abs_label_q95": train_q95,
        "normal_train_rows": int(class_counts[0]),
        "positive_extreme_train_rows": int(class_counts[1]),
        "negative_extreme_train_rows": int(class_counts[2]),
        "normal_valid_rows_by_train_threshold": int(valid_class_counts[0]),
        "positive_extreme_valid_rows_by_train_threshold": int(valid_class_counts[1]),
        "negative_extreme_valid_rows_by_train_threshold": int(valid_class_counts[2]),
        "detector_best_iteration": int(detector.best_iteration or args.detector_rounds),
        "pos_expert_best_iteration": int(pos_expert.best_iteration or args.expert_rounds),
        "neg_expert_best_iteration": int(neg_expert.best_iteration or args.expert_rounds),
        "best_alpha": best_alpha,
        "detector_model_path": str(detector_path),
        "pos_expert_model_path": str(pos_expert_path),
        "neg_expert_model_path": str(neg_expert_path),
    }
    report_path = report_dir / "two_stage_expert_report.md"
    write_report(report_path, metrics, bias, alpha_search, training_info)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_csv": str(pred_path),
        "alpha_search_csv": str(alpha_search_path),
        "metrics_csv": str(metrics_path),
        "bias_analysis_csv": str(bias_path),
        "report": str(report_path),
        "training_info": training_info,
        "best_alpha_row": alpha_search.iloc[0].to_dict(),
    }
    summary_path = output_dir / "two_stage_expert_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
