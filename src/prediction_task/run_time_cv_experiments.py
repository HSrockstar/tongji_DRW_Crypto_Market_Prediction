from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import (  # noqa: E402
    DERIVED_FEATURES,
    MARKET_FIELDS,
    add_basic_market_features,
)
from data_preprocessing.preprocess import (  # noqa: E402
    DEFAULT_ROOT,
    TARGET_COL,
    ensure_dir,
    get_feature_columns,
    load_parquet_frame,
    validate_no_missing_or_infinite,
)
from prediction_task.metrics import evaluate_regression  # noqa: E402
from prediction_task.splits import time_series_cv_splits  # noqa: E402


def lgb_pearson_eval(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    labels = dataset.get_label()
    metrics = evaluate_regression(labels, preds)
    return "pearson", metrics["pearson"], True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行时间 CV 和特征组消融实验")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument(
        "--experiment",
        choices=["full_cv", "feature_group_ablation"],
        required=True,
        help="full_cv 跑完整特征 Ridge+LightGBM；feature_group_ablation 跑 Ridge 特征组消融",
    )
    parser.add_argument("--sample-rows", type=int, default=None, help="只读取前 N 行做烟测")
    parser.add_argument("--n-splits", type=int, default=5, help="时间 CV 折数")
    parser.add_argument("--gap-rows", type=int, default=0, help="训练集和验证集之间的 gap 行数")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge 正则强度")
    parser.add_argument("--ridge-solver", default="lsqr", help="Ridge 求解器")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--metric-for-early-stop", choices=["pearson", "rmse"], default="pearson")
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=10.0)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-gain-to-split", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--log-period", type=int, default=100)
    return parser.parse_args()


def get_lgb_params(args: argparse.Namespace) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": "regression",
        "metric": "None" if args.metric_for_early_stop == "pearson" else "rmse",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "min_gain_to_split": args.min_gain_to_split,
        "seed": 42,
        "verbosity": -1,
    }
    if args.num_threads > 0:
        params["num_threads"] = args.num_threads
    return params


def get_anonymous_features(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith("X") and column[1:].isdigit()]


def build_feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    feature_cols = get_feature_columns(data.columns)
    anonymous_cols = get_anonymous_features(feature_cols)
    groups = {
        "public_market": MARKET_FIELDS,
        "derived_market": DERIVED_FEATURES,
        "public_plus_derived": MARKET_FIELDS + DERIVED_FEATURES,
        "anonymous_x": anonymous_cols,
        "anonymous_plus_public": anonymous_cols + MARKET_FIELDS,
        "full": feature_cols,
    }
    for name, columns in groups.items():
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise ValueError(f"特征组 {name} 缺少字段: {missing}")
        if not columns:
            raise ValueError(f"特征组 {name} 为空")
    return groups


def fit_predict_ridge(
    data: pd.DataFrame,
    split: dict[str, object],
    feature_cols: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    train_idx = split["train_idx"]
    valid_idx = split["valid_idx"]
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.ridge_alpha, solver=args.ridge_solver)),
        ]
    )
    model.fit(data.iloc[train_idx][feature_cols], data.iloc[train_idx][TARGET_COL])
    valid_pred = model.predict(data.iloc[valid_idx][feature_cols])
    return valid_pred, {
        "alpha": args.ridge_alpha,
        "solver": args.ridge_solver,
    }


def fit_predict_lgbm(
    data: pd.DataFrame,
    split: dict[str, object],
    feature_cols: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    train_idx = split["train_idx"]
    valid_idx = split["valid_idx"]
    X_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    X_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
    params = get_lgb_params(args)

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_cols, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        feval=lgb_pearson_eval if args.metric_for_early_stop == "pearson" else None,
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(args.log_period),
        ],
    )
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    return valid_pred, {
        "best_iteration": int(model.best_iteration or args.num_boost_round),
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "lambda_l2": args.lambda_l2,
    }


def run_fold(
    data: pd.DataFrame,
    split: dict[str, object],
    *,
    experiment: str,
    feature_group: str,
    feature_cols: list[str],
    model_name: str,
    args: argparse.Namespace,
    generated_at: str,
) -> dict[str, object]:
    print(f"开始训练: experiment={experiment}, group={feature_group}, model={model_name}, fold={split['fold']}")
    valid_idx = split["valid_idx"]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
    if model_name == "ridge":
        valid_pred, model_info = fit_predict_ridge(data, split, feature_cols, args)
    elif model_name == "lightgbm":
        valid_pred, model_info = fit_predict_lgbm(data, split, feature_cols, args)
    else:
        raise ValueError(f"未知模型: {model_name}")

    metrics = evaluate_regression(y_valid, valid_pred)
    result = {
        "experiment": experiment,
        "feature_group": feature_group,
        "model": model_name,
        "sample_rows": args.sample_rows or len(data),
        "n_splits": args.n_splits,
        "gap_rows": args.gap_rows,
        "fold": split["fold"],
        "train_start": split["train_start"],
        "train_end": split["train_end"],
        "valid_start": split["valid_start"],
        "valid_end": split["valid_end"],
        "train_rows": len(split["train_idx"]),
        "valid_rows": len(split["valid_idx"]),
        "feature_count": len(feature_cols),
        "generated_at": generated_at,
        **model_info,
        **metrics,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "experiment",
        "feature_group",
        "model",
        "sample_rows",
        "n_splits",
        "gap_rows",
        "feature_count",
    ]
    return (
        results.groupby(group_cols, dropna=False)
        .agg(
            fold_count=("fold", "count"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
        )
        .reset_index()
        .sort_values(["experiment", "model", "pearson_mean"], ascending=[True, True, False])
    )


def save_cv_plot(results: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    for model_name, model_results in results.groupby("model"):
        model_results = model_results.sort_values("fold")
        plt.plot(model_results["fold"], model_results["pearson"], marker="o", label=model_name)
    plt.xlabel("fold")
    plt.ylabel("pearson")
    plt.title("CV pearson by fold")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_feature_group_plot(summary: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    summary = summary.sort_values("pearson_mean", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(
        summary["feature_group"],
        summary["pearson_mean"],
        yerr=summary["pearson_std"].fillna(0.0),
        capsize=4,
    )
    plt.xlabel("feature group")
    plt.ylabel("pearson mean")
    plt.title("Feature group ablation")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_full_cv(
    data: pd.DataFrame,
    splits: list[dict[str, object]],
    args: argparse.Namespace,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    feature_cols = get_feature_columns(data.columns)
    generated_at = datetime.now().isoformat(timespec="seconds")
    results = []
    for model_name in ["ridge", "lightgbm"]:
        for split in splits:
            results.append(
                run_fold(
                    data,
                    split,
                    experiment="full_cv",
                    feature_group="full",
                    feature_cols=feature_cols,
                    model_name=model_name,
                    args=args,
                    generated_at=generated_at,
                )
            )

    result_frame = pd.DataFrame(results)
    summary = summarize_results(result_frame)
    result_frame.to_csv(output_dir / "cv_results.csv", index=False)
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    save_cv_plot(result_frame, figure_dir / "cv_pearson_by_fold.png")
    print(f"CV 明细已保存: {output_dir / 'cv_results.csv'}")
    print(f"CV 汇总已保存: {output_dir / 'cv_summary.csv'}")
    print(summary.to_string(index=False))


def run_feature_group_ablation(
    data: pd.DataFrame,
    splits: list[dict[str, object]],
    args: argparse.Namespace,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    feature_groups = build_feature_groups(data)
    generated_at = datetime.now().isoformat(timespec="seconds")
    results = []
    for group_name, feature_cols in feature_groups.items():
        for split in splits:
            results.append(
                run_fold(
                    data,
                    split,
                    experiment="feature_group_ablation",
                    feature_group=group_name,
                    feature_cols=feature_cols,
                    model_name="ridge",
                    args=args,
                    generated_at=generated_at,
                )
            )

    result_frame = pd.DataFrame(results)
    summary = summarize_results(result_frame)
    result_frame.to_csv(output_dir / "feature_group_cv_results.csv", index=False)
    summary.to_csv(output_dir / "feature_group_cv_summary.csv", index=False)
    save_feature_group_plot(summary, figure_dir / "feature_group_compare.png")
    print(f"特征组消融明细已保存: {output_dir / 'feature_group_cv_results.csv'}")
    print(f"特征组消融汇总已保存: {output_dir / 'feature_group_cv_summary.csv'}")
    print(summary.to_string(index=False))


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    prediction_figure_dir = ensure_dir(root / "outputs" / "figures" / "prediction_task")
    feature_figure_dir = ensure_dir(root / "outputs" / "figures" / "feature_effectiveness")

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    data = add_basic_market_features(data)
    feature_cols = get_feature_columns(data.columns)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="时间 CV 实验数据")
    splits = time_series_cv_splits(len(data), n_splits=args.n_splits, gap_rows=args.gap_rows)

    if args.experiment == "full_cv":
        run_full_cv(data, splits, args, output_dir, prediction_figure_dir)
    elif args.experiment == "feature_group_ablation":
        run_feature_group_ablation(data, splits, args, output_dir, feature_figure_dir)
    else:
        raise ValueError(f"未知实验: {args.experiment}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
