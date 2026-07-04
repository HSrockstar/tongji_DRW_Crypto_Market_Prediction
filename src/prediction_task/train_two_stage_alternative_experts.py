from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    "two_stage_expert": "Two-stage LightGBM Expert",
    "two_stage_constant_calibration": "Two-stage Constant Calibration",
    "two_stage_ridge_expert": "Two-stage Ridge Expert",
}
PREDICTION_FILES = {
    "ridge": "official_baseline_valid_predictions.csv",
    "lightgbm": "official_lgbm_valid_predictions.csv",
    "weighted_lightgbm": "extreme_volatility_weighted_lgbm_valid_predictions.csv",
    "continuous_weighted_lightgbm": "extreme_volatility_continuous_weighted_lgbm_valid_predictions.csv",
    "catboost": "tree_baseline_catboost_valid_predictions.csv",
    "xgboost": "tree_baseline_xgboost_valid_predictions.csv",
    "tree_blend": "tree_blend_valid_predictions.csv",
    "two_stage_expert": "two_stage_expert_valid_predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="尝试正负常数残差校准和 Ridge 残差专家")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--gap-rows", type=int, default=0, help="训练集和验证集之间的 gap 行数")
    parser.add_argument("--alpha-step", type=float, default=0.05, help="残差修正强度搜索步长")
    parser.add_argument("--alpha-max", type=float, default=1.5, help="残差修正强度搜索上限")
    parser.add_argument("--ridge-alpha", type=float, default=10.0, help="Ridge 残差专家正则强度")
    parser.add_argument("--ridge-solver", default="lsqr", help="Ridge 求解器")
    parser.add_argument("--ridge-correction-clip", type=float, default=5.0, help="Ridge 软门控残差修正项绝对值裁剪上限")
    return parser.parse_args()


def load_training_frame(root: Path, feature_cols: list[str]) -> pd.DataFrame:
    parquet_cols = set(get_parquet_columns(root / "data" / "raw" / "train.parquet"))
    raw_feature_cols = [column for column in feature_cols if column not in DERIVED_FEATURES and column in parquet_cols]
    selected_columns = sorted(set(raw_feature_cols) | set(MARKET_FIELDS) | {TARGET_COL})
    data = load_parquet_frame(root, "train.parquet", include_label=True, selected_columns=selected_columns)
    data = add_basic_market_features(data)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="两阶段替代专家训练数据")
    return data


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


def load_gate_probabilities(output_dir: Path, valid_idx: np.ndarray, y_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gate_path = output_dir / "two_stage_expert_valid_predictions.csv"
    gate_frame = pd.read_csv(gate_path)
    required = {"p_pos_extreme", "p_neg_extreme", "row_index", "y_true"}
    missing = required - set(gate_frame.columns)
    if missing:
        raise ValueError(f"{gate_path} 缺少软门控概率列: {sorted(missing)}")
    if not np.array_equal(gate_frame["row_index"].to_numpy(), valid_idx):
        raise ValueError("两阶段软门控预测与当前验证切分不一致")
    if not np.allclose(gate_frame["y_true"].to_numpy(dtype=np.float64), y_valid, rtol=0, atol=1e-12):
        raise ValueError("两阶段软门控预测 y_true 与当前验证集不一致")
    return (
        gate_frame["p_pos_extreme"].to_numpy(dtype=np.float64),
        gate_frame["p_neg_extreme"].to_numpy(dtype=np.float64),
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


def make_prediction_frame(
    valid_idx: np.ndarray,
    y_valid: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    *,
    base_pred: np.ndarray,
    correction: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": valid_idx,
            "y_true": y_valid,
            "y_pred": y_pred,
            "model": model_name,
            "base_lgbm_pred": base_pred,
            "correction": correction,
            "alpha": alpha,
        }
    )


def train_ridge_expert(
    x_train: pd.DataFrame,
    residual_train: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
    solver: str,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, solver=solver)),
        ]
    )
    model.fit(x_train.loc[mask], residual_train[mask])
    return model


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
    constant_alpha_search: pd.DataFrame,
    ridge_alpha_search: pd.DataFrame,
    training_info: dict[str, object],
) -> None:
    ensure_dir(path.parent)
    focus_models = ["lightgbm", "two_stage_expert", "two_stage_constant_calibration", "two_stage_ridge_expert"]
    focus_metrics = metrics[metrics["model"].isin(focus_models)].copy()
    focus_bias = bias[
        bias["model"].isin(focus_models)
        & bias["slice"].isin([SLICE_POS_EXTREME, SLICE_NEG_EXTREME])
    ].copy()
    lightgbm_extreme = metrics[(metrics["model"] == "lightgbm") & (metrics["group"] == GROUP_EXTREME)].iloc[0]
    constant_extreme = metrics[
        (metrics["model"] == "two_stage_constant_calibration") & (metrics["group"] == GROUP_EXTREME)
    ].iloc[0]
    ridge_extreme = metrics[(metrics["model"] == "two_stage_ridge_expert") & (metrics["group"] == GROUP_EXTREME)].iloc[0]
    content = f"""# 两阶段替代专家模型报告

生成时间：{datetime.now().isoformat(timespec="seconds")}

## 实验设置

- 软门控：复用已有三分类异常检测器输出的 `p_pos_extreme` 与 `p_neg_extreme`。
- 常数校准：分别使用训练集正向异常、负向异常的平均残差。
- Ridge 专家：分别在正向异常、负向异常训练样本上拟合 `label - base_lgbm_pred`。
- 训练集异常阈值：`abs(label)` 95% 分位 `{training_info["train_abs_label_q95"]:.6f}`。
- Ridge 参数：alpha `{training_info["ridge_alpha"]}`，solver `{training_info["ridge_solver"]}`。
- Ridge 修正项裁剪：`[-{training_info["ridge_correction_clip"]}, {training_info["ridge_correction_clip"]}]`。
- 说明：alpha 直接在当前验证集上搜索，用于课程实验分析，不作为严格无偏泛化估计。

## 重点模型指标

{markdown_table(
        focus_metrics,
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

## 正负异常偏差

{markdown_table(
        focus_bias,
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

常数校准：

{markdown_table(
        constant_alpha_search.head(5),
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

Ridge 专家：

{markdown_table(
        ridge_alpha_search.head(5),
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

- 常数校准相对 LightGBM：异常组 RMSE `{constant_extreme["rmse"] - lightgbm_extreme["rmse"]:+.6f}`，MAE `{constant_extreme["mae"] - lightgbm_extreme["mae"]:+.6f}`。
- Ridge 专家相对 LightGBM：异常组 RMSE `{ridge_extreme["rmse"] - lightgbm_extreme["rmse"]:+.6f}`，MAE `{ridge_extreme["mae"] - lightgbm_extreme["mae"]:+.6f}`。
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
    base_best_iteration = int(pd.read_csv(output_dir / "official_lgbm_results.csv").iloc[0]["best_iteration"])
    base_train_pred = base_model.predict(x_train, num_iteration=base_best_iteration)
    base_valid_frame = pd.read_csv(output_dir / "official_lgbm_valid_predictions.csv")
    if not np.array_equal(base_valid_frame["row_index"].to_numpy(), valid_idx):
        raise ValueError("official_lgbm_valid_predictions.csv 与当前验证切分不一致")
    if not np.allclose(base_valid_frame["y_true"].to_numpy(dtype=np.float64), y_valid, rtol=0, atol=1e-12):
        raise ValueError("official_lgbm_valid_predictions.csv 的 y_true 与当前验证集不一致")
    base_valid_pred = base_valid_frame["y_pred"].to_numpy(dtype=np.float64)
    residual_train = y_train - base_train_pred

    train_q95 = float(np.quantile(np.abs(y_train), 0.95))
    pos_train_mask = y_train > train_q95
    neg_train_mask = y_train < -train_q95
    p_pos_valid, p_neg_valid = load_gate_probabilities(output_dir, valid_idx, y_valid)

    pos_bias = float(residual_train[pos_train_mask].mean())
    neg_bias = float(residual_train[neg_train_mask].mean())
    constant_correction = p_pos_valid * pos_bias + p_neg_valid * neg_bias

    pos_ridge = train_ridge_expert(
        x_train,
        residual_train,
        pos_train_mask,
        alpha=args.ridge_alpha,
        solver=args.ridge_solver,
    )
    neg_ridge = train_ridge_expert(
        x_train,
        residual_train,
        neg_train_mask,
        alpha=args.ridge_alpha,
        solver=args.ridge_solver,
    )
    pos_ridge_path = model_dir / "two_stage_pos_ridge_residual_expert.pkl"
    neg_ridge_path = model_dir / "two_stage_neg_ridge_residual_expert.pkl"
    joblib.dump(pos_ridge, pos_ridge_path)
    joblib.dump(neg_ridge, neg_ridge_path)

    pos_ridge_pred = pos_ridge.predict(x_valid)
    neg_ridge_pred = neg_ridge.predict(x_valid)
    ridge_correction_raw = p_pos_valid * pos_ridge_pred + p_neg_valid * neg_ridge_pred
    if args.ridge_correction_clip <= 0:
        ridge_correction = ridge_correction_raw
    else:
        ridge_correction = np.clip(
            ridge_correction_raw,
            -args.ridge_correction_clip,
            args.ridge_correction_clip,
        )

    group_stats = pd.read_csv(output_dir / "extreme_volatility_group_stats.csv")
    extreme_mask = np.abs(y_valid) > float(group_stats["abs_label_q95_threshold"].iloc[0])
    constant_alpha, constant_alpha_search = search_alpha(
        y_valid,
        base_valid_pred,
        constant_correction,
        extreme_mask,
        alpha_step=args.alpha_step,
        alpha_max=args.alpha_max,
    )
    ridge_alpha, ridge_alpha_search = search_alpha(
        y_valid,
        base_valid_pred,
        ridge_correction,
        extreme_mask,
        alpha_step=args.alpha_step,
        alpha_max=args.alpha_max,
    )
    constant_pred = base_valid_pred + constant_alpha * constant_correction
    ridge_pred = base_valid_pred + ridge_alpha * ridge_correction

    constant_frame = make_prediction_frame(
        valid_idx,
        y_valid,
        constant_pred,
        "two_stage_constant_calibration",
        base_pred=base_valid_pred,
        correction=constant_correction,
        alpha=constant_alpha,
    )
    ridge_frame = make_prediction_frame(
        valid_idx,
        y_valid,
        ridge_pred,
        "two_stage_ridge_expert",
        base_pred=base_valid_pred,
        correction=ridge_correction,
        alpha=ridge_alpha,
    )
    constant_pred_path = output_dir / "two_stage_constant_calibration_valid_predictions.csv"
    ridge_pred_path = output_dir / "two_stage_ridge_expert_valid_predictions.csv"
    constant_frame.to_csv(constant_pred_path, index=False)
    ridge_frame.to_csv(ridge_pred_path, index=False)

    constant_alpha_path = output_dir / "two_stage_constant_calibration_alpha_search.csv"
    ridge_alpha_path = output_dir / "two_stage_ridge_expert_alpha_search.csv"
    constant_alpha_search.to_csv(constant_alpha_path, index=False, encoding="utf-8-sig")
    ridge_alpha_search.to_csv(ridge_alpha_path, index=False, encoding="utf-8-sig")

    frames = load_prediction_frames(output_dir)
    frames["two_stage_constant_calibration"] = constant_frame[["row_index", "y_true", "y_pred", "model"]].copy()
    frames["two_stage_ridge_expert"] = ridge_frame[["row_index", "y_true", "y_pred", "model"]].copy()
    metrics = compute_metrics(frames, group_stats)
    bias = compute_bias_analysis(frames, group_stats)

    metrics_path = output_dir / "two_stage_alternative_experts_metrics.csv"
    bias_path = output_dir / "two_stage_alternative_experts_bias_analysis.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    bias.to_csv(bias_path, index=False, encoding="utf-8-sig")

    training_info: dict[str, object] = {
        "train_abs_label_q95": train_q95,
        "positive_extreme_train_rows": int(pos_train_mask.sum()),
        "negative_extreme_train_rows": int(neg_train_mask.sum()),
        "pos_bias": pos_bias,
        "neg_bias": neg_bias,
        "constant_best_alpha": constant_alpha,
        "ridge_best_alpha": ridge_alpha,
        "ridge_alpha": args.ridge_alpha,
        "ridge_solver": args.ridge_solver,
        "ridge_correction_clip": args.ridge_correction_clip,
        "ridge_correction_raw_min": float(np.min(ridge_correction_raw)),
        "ridge_correction_raw_max": float(np.max(ridge_correction_raw)),
        "ridge_correction_min": float(np.min(ridge_correction)),
        "ridge_correction_max": float(np.max(ridge_correction)),
        "pos_ridge_model_path": str(pos_ridge_path),
        "neg_ridge_model_path": str(neg_ridge_path),
    }
    report_path = report_dir / "two_stage_alternative_experts_report.md"
    write_report(report_path, metrics, bias, constant_alpha_search, ridge_alpha_search, training_info)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "constant_prediction_csv": str(constant_pred_path),
        "ridge_prediction_csv": str(ridge_pred_path),
        "constant_alpha_search_csv": str(constant_alpha_path),
        "ridge_alpha_search_csv": str(ridge_alpha_path),
        "metrics_csv": str(metrics_path),
        "bias_analysis_csv": str(bias_path),
        "report": str(report_path),
        "training_info": training_info,
        "constant_best_alpha_row": constant_alpha_search.iloc[0].to_dict(),
        "ridge_best_alpha_row": ridge_alpha_search.iloc[0].to_dict(),
    }
    summary_path = output_dir / "two_stage_alternative_experts_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
