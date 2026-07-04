from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import DERIVED_FEATURES, MARKET_FIELDS, add_basic_market_features
from data_preprocessing.preprocess import (
    DEFAULT_ROOT,
    TARGET_COL,
    ensure_dir,
    get_feature_columns,
    load_json,
    load_parquet_frame,
    save_json,
    validate_no_missing_or_infinite,
)
from feature_effectiveness.clustering import (
    compute_feature_target_correlation,
    filter_low_signal_features,
    get_anonymous_feature_names,
    select_medoid_features,
)
from feature_effectiveness.stable_features import (
    collect_fold_top_features,
    importance_summary,
    select_stable_features,
    train_lgbm_fold_importance,
)
from prediction_task.metrics import evaluate_regression
from prediction_task.run_time_cv_experiments import fit_predict_lgbm, fit_predict_ridge
from prediction_task.splits import purged_group_time_series_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冠军风格特征筛选管线 + CV 对比")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--sample-rows", type=int, default=None, help="烟测行数")
    parser.add_argument("--n-groups", type=int, default=6, help="purged group CV 组数")
    parser.add_argument("--gap-groups", type=int, default=1, help="验证组前的 gap 组数")
    parser.add_argument("--corr-threshold", type=float, default=0.6, help="特征聚类相关性阈值")
    parser.add_argument("--target-corr-threshold", type=float, default=1e-4, help="低信号特征阈值")
    parser.add_argument("--cluster-sample-rows", type=int, default=100_000, help="聚类相关性采样行数")
    parser.add_argument("--top-k-importance", type=int, default=20, help="每折 SHAP/importance top-k")
    parser.add_argument("--min-stable-folds", type=int, default=3, help="稳定特征最少出现折数")
    parser.add_argument("--include-market", action="store_true", default=True, help="稳定特征集加入市场特征")
    parser.add_argument("--skip-cv-compare", action="store_true", help="仅生成特征列表，跳过 CV 对比")
    parser.add_argument(
        "--only-cv-compare",
        action="store_true",
        help="跳过特征筛选，直接读取已保存的 feature_set_definitions.json 做 CV 对比",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
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


def build_feature_sets(
    all_features: list[str],
    medoid_x: list[str],
    filtered_x: list[str],
    stable_x: list[str],
    *,
    include_market: bool,
) -> dict[str, list[str]]:
    market_block = MARKET_FIELDS + DERIVED_FEATURES if include_market else []
    return {
        "full": all_features,
        "medoid_x_plus_market": sorted(set(medoid_x + market_block)),
        "filtered_x_plus_market": sorted(set(filtered_x + market_block)),
        "stable_x_plus_market": sorted(set(stable_x + market_block)),
    }


def run_cv_for_feature_set(
    data: pd.DataFrame,
    splits: list[dict[str, object]],
    feature_cols: list[str],
    feature_set_name: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    generated_at = datetime.now().isoformat(timespec="seconds")
    results: list[dict[str, object]] = []
    for split in splits:
        for model_name in ["ridge", "lightgbm"]:
            valid_idx = split["valid_idx"]
            y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
            if model_name == "ridge":
                valid_pred, model_info = fit_predict_ridge(
                    data,
                    split,
                    feature_cols,
                    args,
                )
            else:
                valid_pred, model_info = fit_predict_lgbm(
                    data,
                    split,
                    feature_cols,
                    args,
                )
            metrics = evaluate_regression(y_valid, valid_pred)
            results.append(
                {
                    "experiment": "feature_pipeline_cv",
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "sample_rows": args.sample_rows or len(data),
                    "n_groups": args.n_groups,
                    "gap_groups": args.gap_groups,
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
            )
            print(
                f"CV {feature_set_name} {model_name} fold={split['fold']} "
                f"pearson={metrics['pearson']:.6f}"
            )
    return results


def save_feature_set_compare_plot(summary: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    lgbm_summary = summary[summary["model"] == "lightgbm"].sort_values("pearson_mean", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(
        lgbm_summary["feature_set"],
        lgbm_summary["pearson_mean"],
        yerr=lgbm_summary["pearson_std"].fillna(0.0),
        capsize=4,
    )
    plt.xlabel("feature set")
    plt.ylabel("pearson mean")
    plt.title("Feature pipeline CV (LightGBM)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    figure_dir = ensure_dir(root / "outputs" / "figures" / "feature_effectiveness")
    model_dir = ensure_dir(root / "models")

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    data = add_basic_market_features(data)
    all_features = get_feature_columns(data.columns)
    anonymous_features = get_anonymous_feature_names(all_features)
    validate_no_missing_or_infinite(data, all_features + [TARGET_COL], context="特征管线数据")

    splits = purged_group_time_series_splits(
        len(data),
        n_groups=args.n_groups,
        gap_groups=args.gap_groups,
    )

    if args.only_cv_compare:
        feature_sets_path = output_dir / "feature_set_definitions.json"
        if not feature_sets_path.is_file():
            raise FileNotFoundError(f"未找到 {feature_sets_path}，请先完整运行特征管线")
        feature_sets = load_json(feature_sets_path)
        print(f"跳过特征筛选，直接对比 {len(feature_sets)} 组特征集")
    else:
        print(f"Step 1/4: 对 {len(anonymous_features)} 个匿名特征做相关性聚类 (threshold={args.corr_threshold})")
        medoid_x, cluster_rows = select_medoid_features(
            data,
            anonymous_features,
            threshold=args.corr_threshold,
            sample_rows=args.cluster_sample_rows,
        )
        cluster_frame = pd.DataFrame(cluster_rows)
        cluster_frame.to_csv(output_dir / "feature_clusters.csv", index=False)
        save_json(
            {
                "corr_threshold": args.corr_threshold,
                "cluster_count": len(set(cluster_frame["cluster_id"])),
                "medoid_features": medoid_x,
                "medoid_count": len(medoid_x),
            },
            output_dir / "medoid_features.json",
        )
        print(f"  medoid 特征数: {len(medoid_x)}")

        print(f"Step 2/4: 目标相关性粗筛 (threshold={args.target_corr_threshold})")
        target_corr = compute_feature_target_correlation(
            data,
            medoid_x,
            TARGET_COL,
            sample_rows=args.sample_rows,
        )
        target_corr.sort_values(key=lambda series: series.abs(), ascending=False).to_frame(
            "target_corr"
        ).to_csv(output_dir / "feature_target_corr.csv")
        filtered_x, dropped_x = filter_low_signal_features(
            target_corr,
            threshold=args.target_corr_threshold,
        )
        save_json(
            {
                "target_corr_threshold": args.target_corr_threshold,
                "kept_features": filtered_x,
                "dropped_features": dropped_x,
                "kept_count": len(filtered_x),
                "dropped_count": len(dropped_x),
            },
            output_dir / "low_signal_features.json",
        )
        print(f"  保留 {len(filtered_x)} 个，低信号 {len(dropped_x)} 个")

        print(
            f"Step 3/4: purged CV ({args.n_groups} groups, gap={args.gap_groups}) "
            f"→ {len(splits)} 折，收集 importance"
        )

        fold_importance_parts: list[pd.DataFrame] = []
        candidate_features = sorted(set(filtered_x + MARKET_FIELDS + DERIVED_FEATURES))
        for split in splits:
            train_idx = split["train_idx"]
            valid_idx = split["valid_idx"]
            X_train = data.iloc[train_idx][candidate_features]
            y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
            X_valid = data.iloc[valid_idx][candidate_features]
            y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
            _, importance_frame = train_lgbm_fold_importance(
                X_train,
                y_train,
                X_valid,
                y_valid,
                feature_names=candidate_features,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
                min_data_in_leaf=args.min_data_in_leaf,
                lambda_l2=args.lambda_l2,
            )
            fold_importance_parts.append(
                collect_fold_top_features(
                    importance_frame, fold=int(split["fold"]), top_k=args.top_k_importance
                )
            )

        fold_importance = pd.concat(fold_importance_parts, ignore_index=True)
        fold_importance.to_csv(output_dir / "fold_feature_importance.csv", index=False)
        importance_overview = importance_summary(fold_importance)
        importance_overview.to_csv(output_dir / "feature_importance_summary.csv", index=False)

        stable_x = [
            feature
            for feature in select_stable_features(
                fold_importance,
                top_k=args.top_k_importance,
                min_folds=args.min_stable_folds,
            )
            if feature.startswith("X")
        ]
        if not stable_x:
            stable_x = (
                importance_overview[importance_overview["feature"].str.startswith("X")]
                .head(30)["feature"]
                .tolist()
            )

        save_json(
            {
                "top_k_importance": args.top_k_importance,
                "min_stable_folds": args.min_stable_folds,
                "stable_anonymous_features": stable_x,
                "stable_anonymous_count": len(stable_x),
            },
            output_dir / "stable_features.json",
        )
        print(f"  稳定匿名特征数: {len(stable_x)}")

        feature_sets = build_feature_sets(
            all_features,
            medoid_x,
            filtered_x,
            stable_x,
            include_market=args.include_market,
        )
        save_json({name: cols for name, cols in feature_sets.items()}, output_dir / "feature_set_definitions.json")

        final_features = feature_sets["stable_x_plus_market"]
        save_json({"feature_columns": final_features}, model_dir / "selected_features.json")
        print(f"Step 4/4: 最终精选特征集 stable_x_plus_market，共 {len(final_features)} 个")

    if args.skip_cv_compare:
        print("已跳过 CV 对比")
        return 0

    cv_results: list[dict[str, object]] = []
    for set_name, feature_cols in feature_sets.items():
        cv_results.extend(run_cv_for_feature_set(data, splits, feature_cols, set_name, args))

    cv_frame = pd.DataFrame(cv_results)
    cv_summary = (
        cv_frame.groupby(
            ["experiment", "feature_set", "model", "sample_rows", "n_groups", "gap_groups", "feature_count"],
            dropna=False,
        )
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
        .sort_values(["model", "pearson_mean"], ascending=[True, False])
    )
    cv_frame.to_csv(output_dir / "feature_pipeline_cv_results.csv", index=False)
    cv_summary.to_csv(output_dir / "feature_pipeline_cv_summary.csv", index=False)
    save_feature_set_compare_plot(cv_summary, figure_dir / "feature_pipeline_compare.png")
    print(cv_summary.to_string(index=False))
    print(f"CV 对比已保存: {output_dir / 'feature_pipeline_cv_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
