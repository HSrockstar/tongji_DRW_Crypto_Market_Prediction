from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

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
MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "catboost": "CatBoost baseline",
    "xgboost": "XGBoost baseline",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 LightGBM、CatBoost、XGBoost 在整体和异常组上的验证表现")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--gap-rows", type=int, default=0, help="训练集和验证集之间的 gap 行数")
    parser.add_argument("--cat-iterations", type=int, default=3000, help="CatBoost 最大迭代轮数")
    parser.add_argument("--xgb-estimators", type=int, default=3000, help="XGBoost 最大树数")
    parser.add_argument("--early-stopping-rounds", type=int, default=200, help="早停轮数")
    parser.add_argument("--log-period", type=int, default=200, help="训练日志间隔")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，读取已有 CatBoost/XGBoost 预测")
    return parser.parse_args()


def load_training_frame(root: Path, feature_cols: list[str]) -> pd.DataFrame:
    parquet_cols = set(get_parquet_columns(root / "data" / "raw" / "train.parquet"))
    raw_feature_cols = [column for column in feature_cols if column not in DERIVED_FEATURES and column in parquet_cols]
    selected_columns = sorted(set(raw_feature_cols) | set(MARKET_FIELDS) | {TARGET_COL})
    data = load_parquet_frame(root, "train.parquet", include_label=True, selected_columns=selected_columns)
    data = add_basic_market_features(data)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="树模型 baseline 对比训练数据")
    return data


def save_prediction(path: Path, row_index: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, model: str) -> pd.DataFrame:
    prediction = pd.DataFrame(
        {
            "row_index": row_index,
            "y_true": y_true,
            "y_pred": np.asarray(y_pred, dtype=np.float64),
            "model": model,
        }
    )
    prediction.to_csv(path, index=False)
    return prediction


def train_catboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[CatBoostRegressor, np.ndarray, dict[str, float]]:
    model = CatBoostRegressor(
        iterations=args.cat_iterations,
        learning_rate=0.01,
        depth=6,
        l2_leaf_reg=50.0,
        random_strength=1.0,
        rsm=0.7,
        min_data_in_leaf=500,
        early_stopping_rounds=args.early_stopping_rounds,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=args.seed,
        verbose=args.log_period,
    )
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
    valid_pred = model.predict(x_valid).astype(np.float64)
    return model, valid_pred, evaluate_regression(y_valid, valid_pred)


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[XGBRegressor, np.ndarray, dict[str, float]]:
    model = XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=500,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=50.0,
        n_estimators=args.xgb_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=1,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=args.log_period)
    valid_pred = model.predict(x_valid).astype(np.float64)
    return model, valid_pred, evaluate_regression(y_valid, valid_pred)


def validate_prediction_alignment(frames: dict[str, pd.DataFrame]) -> np.ndarray:
    reference = next(iter(frames.values()))
    y_true = reference["y_true"].to_numpy(dtype=np.float64)
    for model_name, frame in frames.items():
        if len(frame) != len(reference):
            raise ValueError(f"{model_name} 验证预测行数不一致")
        if not frame["row_index"].equals(reference["row_index"]):
            raise ValueError(f"{model_name} 验证预测 row_index 不一致")
        if not np.allclose(frame["y_true"].to_numpy(dtype=np.float64), y_true, rtol=0, atol=1e-12):
            raise ValueError(f"{model_name} 验证预测 y_true 不一致")
    return y_true


def build_group_masks(y_true: np.ndarray, group_stats: pd.DataFrame) -> list[tuple[str, np.ndarray, str]]:
    q95 = float(group_stats["abs_label_q95_threshold"].iloc[0])
    abs_label = np.abs(y_true)
    return [
        (GROUP_OVERALL, np.ones(len(y_true), dtype=bool), "all validation samples"),
        (GROUP_EXTREME, abs_label > q95, f"abs(label) > {q95:.12g}"),
    ]


def compute_metrics(frames: dict[str, pd.DataFrame], group_stats: pd.DataFrame) -> pd.DataFrame:
    y_true = validate_prediction_alignment(frames)
    groups = build_group_masks(y_true, group_stats)
    rows: list[dict[str, object]] = []
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


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "model_label",
        "group",
        "sample_ratio",
        "pearson",
        "rmse",
        "mae",
        "squared_error_contribution",
        "absolute_error_contribution",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in frame[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_label"]),
                    str(row["group"]),
                    f"{row['sample_ratio']:.2%}",
                    f"{row['pearson']:.6f}",
                    f"{row['rmse']:.6f}",
                    f"{row['mae']:.6f}",
                    f"{row['squared_error_contribution']:.2%}",
                    f"{row['absolute_error_contribution']:.2%}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(path: Path, metrics: pd.DataFrame, training_info: dict[str, object]) -> None:
    ensure_dir(path.parent)
    content = f"""# CatBoost / XGBoost / LightGBM 整体与异常组对比

生成时间：{datetime.now().isoformat(timespec="seconds")}

## 实验设置

- 数据划分：按时间顺序前 80% 训练、后 20% 验证。
- 特征列：复用 `models/official_lgbm_features.json`。
- LightGBM：复用 `outputs/experiments/official_lgbm_valid_predictions.csv`。
- 异常组定义：验证集 `abs(label)` 大于 95% 分位数。
- CatBoost 最佳迭代：`{training_info.get("catboost_best_iteration", "")}`。
- XGBoost 最佳迭代：`{training_info.get("xgboost_best_iteration", "")}`。

## 指标对比

{markdown_table(metrics)}

## 简要结论

- 整体 Pearson 最高模型：`{metrics[metrics["group"] == GROUP_OVERALL].sort_values("pearson", ascending=False).iloc[0]["model_label"]}`。
- 异常组 RMSE 最低模型：`{metrics[metrics["group"] == GROUP_EXTREME].sort_values("rmse", ascending=True).iloc[0]["model_label"]}`。
- 异常组 Pearson 最高模型：`{metrics[metrics["group"] == GROUP_EXTREME].sort_values("pearson", ascending=False).iloc[0]["model_label"]}`。
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    report_dir = ensure_dir(root / "outputs" / "reports")
    model_dir = ensure_dir(root / "models")

    cat_pred_path = output_dir / "tree_baseline_catboost_valid_predictions.csv"
    xgb_pred_path = output_dir / "tree_baseline_xgboost_valid_predictions.csv"
    training_info: dict[str, object] = {}

    if args.skip_train:
        if not cat_pred_path.is_file() or not xgb_pred_path.is_file():
            raise FileNotFoundError("未找到已有 CatBoost/XGBoost baseline 预测，不能使用 --skip-train")
        cat_pred = pd.read_csv(cat_pred_path)
        xgb_pred = pd.read_csv(xgb_pred_path)
    else:
        feature_cols = load_json(model_dir / "official_lgbm_features.json")["feature_columns"]
        data = load_training_frame(root, feature_cols)
        train_idx, valid_idx = time_order_split(len(data), valid_fraction=args.valid_fraction, gap_rows=args.gap_rows)
        x_train = data.iloc[train_idx][feature_cols]
        y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
        x_valid = data.iloc[valid_idx][feature_cols]
        y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)

        cat_model, cat_valid_pred, cat_metrics = train_catboost(x_train, y_train, x_valid, y_valid, args)
        cat_model_path = model_dir / "tree_baseline_catboost.cbm"
        cat_model.save_model(cat_model_path)
        cat_pred = save_prediction(cat_pred_path, valid_idx, y_valid, cat_valid_pred, "catboost")

        xgb_model, xgb_valid_pred, xgb_metrics = train_xgboost(x_train, y_train, x_valid, y_valid, args)
        xgb_model_path = model_dir / "tree_baseline_xgboost.json"
        xgb_model.save_model(xgb_model_path)
        xgb_pred = save_prediction(xgb_pred_path, valid_idx, y_valid, xgb_valid_pred, "xgboost")

        training_info = {
            "feature_count": len(feature_cols),
            "train_rows": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
            "catboost_model_path": str(cat_model_path),
            "xgboost_model_path": str(xgb_model_path),
            "catboost_best_iteration": int(cat_model.get_best_iteration()),
            "xgboost_best_iteration": int(getattr(xgb_model, "best_iteration", args.xgb_estimators)),
            "catboost_overall_metrics": cat_metrics,
            "xgboost_overall_metrics": xgb_metrics,
        }

    frames = {
        "lightgbm": pd.read_csv(output_dir / "official_lgbm_valid_predictions.csv"),
        "catboost": cat_pred,
        "xgboost": xgb_pred,
    }
    group_stats = pd.read_csv(output_dir / "extreme_volatility_group_stats.csv")
    metrics = compute_metrics(frames, group_stats)

    metrics_path = output_dir / "tree_baseline_overall_extreme_metrics.csv"
    summary_path = output_dir / "tree_baseline_overall_extreme_summary.json"
    report_path = report_dir / "tree_baseline_overall_extreme_compare.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    write_report(report_path, metrics, training_info)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics_csv": str(metrics_path),
        "report": str(report_path),
        "catboost_prediction_csv": str(cat_pred_path),
        "xgboost_prediction_csv": str(xgb_pred_path),
        "training_info": training_info,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
