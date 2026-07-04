from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SRC_DIR = Path(__file__).resolve().parents[1]
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import add_basic_market_features
from data_preprocessing.preprocess import DEFAULT_ROOT, TARGET_COL, ensure_dir, get_feature_columns, load_json, load_parquet_frame, save_json, validate_no_missing_or_infinite
from prediction_task.metrics import evaluate_regression
from prediction_task.splits import time_order_split
from prediction_task.train_lgbm import lgb_pearson_eval

DEFAULT_ROOT_PATH = Path(DEFAULT_ROOT)
EXPERIMENTS_DIR = "outputs/experiments"
FIGURES_DIR = "outputs/figures/prediction_task"

PHASE12_MODELS = [
    {
        "key": "phase1a_ensemble",
        "label": "阶段1a 集成 预测",
        "color": "#55A868",
    },
    {
        "key": "phase1b_tuned_lgbm",
        "label": "阶段1b 调参 LGBM 预测",
        "color": "#C44E52",
    },
    {
        "key": "phase1_tuned_ensemble",
        "label": "阶段1 再集成 预测",
        "color": "#8172B3",
    },
    {
        "key": "phase2_combo_lgbm",
        "label": "阶段2 组合特征 预测",
        "color": "#CCB974",
    },
]

DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "None",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 10.0,
    "min_gain_to_split": 0.0,
    "seed": 42,
    "verbosity": -1,
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_valid_predictions(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[["row_index", "y_true", "y_pred"]].copy()
    frame["model"] = model_name
    return frame


def export_default_lgbm_valid_predictions(root: Path) -> pd.DataFrame:
    output_path = root / EXPERIMENTS_DIR / "default_lgbm_valid_predictions.csv"
    if output_path.is_file():
        return read_valid_predictions(output_path, "default_lgbm")

    data = load_parquet_frame(root, "train.parquet", include_label=True)
    data = add_basic_market_features(data)
    feature_cols = get_feature_columns(data.columns)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="default LGBM holdout export")

    train_idx, valid_idx = time_order_split(len(data), valid_fraction=0.2, gap_rows=0)
    X_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    X_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_cols, reference=train_set)
    model = lgb.train(
        DEFAULT_LGBM_PARAMS,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        feval=lgb_pearson_eval,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    metrics = evaluate_regression(y_valid, valid_pred)
    frame = pd.DataFrame(
        {
            "row_index": valid_idx,
            "y_true": y_valid,
            "y_pred": valid_pred,
            "model": "default_lgbm",
        }
    )
    ensure_dir(output_path.parent)
    frame.to_csv(output_path, index=False)
    save_json({"params": DEFAULT_LGBM_PARAMS, **metrics}, root / EXPERIMENTS_DIR / "default_lgbm_valid_export.json")
    print(f"已导出默认 LGBM 验证预测: {output_path} pearson={metrics['pearson']:.6f}")
    return frame


def build_blend_predictions(
    ridge: pd.DataFrame,
    lgbm: pd.DataFrame,
    model_name: str,
    ridge_weight: float,
) -> pd.DataFrame:
    merged = ridge.merge(lgbm, on=["row_index", "y_true"], suffixes=("_ridge", "_lgbm"))
    blend = merged.copy()
    blend["y_pred"] = ridge_weight * merged["y_pred_ridge"] + (1.0 - ridge_weight) * merged["y_pred_lgbm"]
    return blend[["row_index", "y_true", "y_pred"]].assign(model=model_name)


def load_phase12_valid_predictions(root: Path) -> pd.DataFrame:
    experiments = root / EXPERIMENTS_DIR
    ridge = read_valid_predictions(experiments / "official_baseline_valid_predictions.csv", "ridge")
    tuned_lgbm = read_valid_predictions(experiments / "official_lgbm_valid_predictions.csv", "tuned_lgbm")
    default_lgbm = export_default_lgbm_valid_predictions(root)
    combo = read_valid_predictions(experiments / "synthesized_lgbm_valid_predictions.csv", "synthesized_lgbm")

    meta_path = experiments / "ensemble_meta.json"
    ridge_weight = 0.5
    if meta_path.is_file():
        ridge_weight = float(load_json(meta_path)["ridge_weight"])

    phase1a = build_blend_predictions(ridge, default_lgbm, "phase1a_ensemble", 0.5)
    phase1_tuned = build_blend_predictions(ridge, tuned_lgbm, "phase1_tuned_ensemble", ridge_weight)
    phase1b = tuned_lgbm.copy()
    phase1b["model"] = "phase1b_tuned_lgbm"
    phase2 = combo.copy()
    phase2["model"] = "phase2_combo_lgbm"

    combined = pd.concat([phase1a, phase1b, phase1_tuned, phase2], ignore_index=True)
    cache_path = experiments / "phase12_valid_predictions.csv"
    combined.to_csv(cache_path, index=False)
    print(f"已保存阶段验证预测汇总: {cache_path}")
    return combined


def plot_phase12_valid_distribution(root: Path, output_dir: Path) -> Path:
    data = load_phase12_valid_predictions(root)
    true_data = data.drop_duplicates(subset=["row_index"])
    model_meta = {item["key"]: item for item in PHASE12_MODELS}

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(
        data=true_data,
        x="y_true",
        ax=ax,
        label="真实值",
        color="#222222",
        linestyle="--",
        linewidth=2.2,
        common_norm=False,
    )
    for model_key in [item["key"] for item in PHASE12_MODELS]:
        model_data = data[data["model"] == model_key]
        meta = model_meta[model_key]
        sns.kdeplot(
            data=model_data,
            x="y_pred",
            ax=ax,
            label=meta["label"],
            color=meta["color"],
            linewidth=1.8,
            common_norm=False,
        )

    ax.set_title("验证集真实值与预测值分布")
    ax.set_xlabel("数值")
    ax.set_ylabel("密度")
    ax.legend(title="曲线")
    fig.tight_layout()
    out_path = output_dir / "phase12_validation_prediction_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成阶段1+2 验证集预测分布对比图")
    parser.add_argument("--root", default=str(DEFAULT_ROOT_PATH), help="项目根目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / FIGURES_DIR)
    setup_style()
    out_path = plot_phase12_valid_distribution(root, output_dir)
    print(f"阶段验证分布图已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
