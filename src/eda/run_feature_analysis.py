from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import DERIVED_FEATURES, MARKET_FIELDS, add_basic_market_features
from data_preprocessing.preprocess import (
    DEFAULT_ROOT,
    TARGET_COL,
    ensure_dir,
    get_feature_columns,
    get_parquet_columns,
    load_parquet_frame,
    raw_path,
    save_json,
)
from feature_effectiveness.clustering import (
    build_correlation_clusters,
    compute_abs_correlation_matrix,
    compute_feature_target_correlation,
    get_anonymous_feature_names,
)
from prediction_task.make_submission import load_lgbm_model
from prediction_task.splits import time_order_split


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def feature_group(name: str) -> str:
    if name in MARKET_FIELDS:
        return "public_market"
    if name in DERIVED_FEATURES:
        return "derived_market"
    if name.startswith("X") and name[1:].isdigit():
        return "anonymous_x"
    return "other"


def analyze_label(data: pd.DataFrame, output_dir: Path, figures_dir: Path) -> dict:
    train_idx, valid_idx = time_order_split(len(data), valid_fraction=0.2, gap_rows=0)
    y_all = data[TARGET_COL].astype(np.float64)
    y_train = y_all.iloc[train_idx]
    y_valid = y_all.iloc[valid_idx]

    def summarize(series: pd.Series, split_name: str) -> dict[str, float | int | str]:
        arr = series.to_numpy(dtype=np.float64)
        return {
            "split": split_name,
            "count": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "q01": float(np.quantile(arr, 0.01)),
            "q50": float(np.quantile(arr, 0.50)),
            "q99": float(np.quantile(arr, 0.99)),
            "pct_abs_gt_1": float(np.mean(np.abs(arr) > 1.0)),
            "pct_abs_gt_3": float(np.mean(np.abs(arr) > 3.0)),
        }

    rows = [summarize(y_all, "full"), summarize(y_train, "train_80pct"), summarize(y_valid, "holdout_20pct")]
    stats_path = output_dir / "label_stats.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(y_all, bins=80, kde=True, ax=axes[0])
    axes[0].set_title("label 全量分布")
    axes[0].set_xlabel("label")

    plot_df = pd.DataFrame(
        {
            "label": pd.concat([y_train, y_valid], ignore_index=True),
            "split": ["train_80pct"] * len(y_train) + ["holdout_20pct"] * len(y_valid),
        }
    )
    sns.kdeplot(data=plot_df, x="label", hue="split", ax=axes[1], common_norm=False)
    axes[1].set_title("训练段 vs Holdout label 分布")
    axes[1].set_xlabel("label")
    fig.tight_layout()
    fig.savefig(figures_dir / "label_distribution.png", dpi=150)
    plt.close(fig)

    return {
        "label_stats_csv": str(stats_path),
        "holdout_std_over_train_std": float(y_valid.std() / y_train.std()) if y_train.std() > 0 else 0.0,
    }


def analyze_feature_label_correlation(
    data: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    figures_dir: Path,
    *,
    sample_rows: int | None,
) -> pd.DataFrame:
    corr = compute_feature_target_correlation(data, feature_cols, TARGET_COL, sample_rows=sample_rows)
    frame = corr.rename("pearson_with_label").reset_index()
    frame.columns = ["feature", "pearson_with_label"]
    frame["abs_pearson"] = frame["pearson_with_label"].abs()
    frame["feature_group"] = frame["feature"].map(feature_group)
    frame = frame.sort_values("abs_pearson", ascending=False).reset_index(drop=True)
    frame.to_csv(output_dir / "feature_label_corr.csv", index=False)

    group_summary = (
        frame.groupby("feature_group")["abs_pearson"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .sort_values("max", ascending=False)
    )
    group_summary.to_csv(output_dir / "feature_label_corr_by_group.csv", index=False)

    top = frame.head(30)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top, y="feature", x="abs_pearson", hue="feature_group", dodge=False)
    plt.title("与 label 绝对 Pearson 相关性 Top-30")
    plt.xlabel("|Pearson(label)|")
    plt.ylabel("特征")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_label_corr_top30.png", dpi=150)
    plt.close()

    return frame


def analyze_x_clusters(
    data: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
    *,
    sample_rows: int,
    threshold: float,
) -> dict:
    anonymous = get_anonymous_feature_names(list(data.columns))
    corr_matrix = compute_abs_correlation_matrix(data, anonymous, sample_rows=sample_rows)
    clusters = build_correlation_clusters(corr_matrix, threshold=threshold)
    sizes = [len(cluster) for cluster in clusters]

    cluster_rows = []
    for cluster_id, cluster in enumerate(clusters, start=1):
        for feature in cluster:
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "feature": feature,
                    "cluster_size": len(cluster),
                }
            )
    pd.DataFrame(cluster_rows).to_csv(output_dir / "x_correlation_clusters.csv", index=False)

    size_frame = pd.DataFrame({"cluster_size": sizes})
    size_frame.to_csv(output_dir / "x_cluster_size_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.histplot(sizes, bins=min(40, max(10, len(set(sizes)))), kde=False)
    plt.title(f"匿名特征共线簇大小分布 (|corr|>={threshold}, sample={sample_rows})")
    plt.xlabel("簇大小")
    plt.ylabel("簇数量")
    plt.tight_layout()
    plt.savefig(figures_dir / "x_cluster_size_distribution.png", dpi=150)
    plt.close()

    return {
        "anonymous_feature_count": len(anonymous),
        "cluster_count": len(clusters),
        "cluster_size_min": int(min(sizes)),
        "cluster_size_median": float(np.median(sizes)),
        "cluster_size_max": int(max(sizes)),
        "cluster_size_mean": float(np.mean(sizes)),
        "singleton_cluster_count": int(sum(size == 1 for size in sizes)),
    }


def analyze_lgbm_importance(
    root: Path,
    corr_frame: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    model_path = root / "models" / "official_lgbm.txt"
    if not model_path.is_file():
        print(f"未找到 LGBM 模型，跳过 importance: {model_path}")
        return pd.DataFrame()

    model = load_lgbm_model(model_path)
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    frame = pd.DataFrame({"feature": feature_names, "gain_importance": importance})
    frame["feature_group"] = frame["feature"].map(feature_group)
    frame = frame.sort_values("gain_importance", ascending=False).reset_index(drop=True)
    frame.to_csv(output_dir / "lgbm_gain_importance.csv", index=False)

    top = frame.head(30)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top, y="feature", x="gain_importance", hue="feature_group", dodge=False)
    plt.title("调参 LGBM Gain Importance Top-30")
    plt.xlabel("gain importance")
    plt.ylabel("特征")
    plt.tight_layout()
    plt.savefig(figures_dir / "lgbm_importance_top30.png", dpi=150)
    plt.close()

    merged = frame.merge(corr_frame[["feature", "abs_pearson"]], on="feature", how="left")
    merged.to_csv(output_dir / "importance_vs_label_corr.csv", index=False)

    plt.figure(figsize=(7, 6))
    plot_data = merged.head(100)
    sns.scatterplot(data=plot_data, x="abs_pearson", y="gain_importance", hue="feature_group", alpha=0.8)
    plt.title("Top-100 特征: |corr(label)| vs LGBM importance")
    plt.xlabel("|Pearson(label)|")
    plt.ylabel("gain importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "importance_vs_corr_scatter.png", dpi=150)
    plt.close()

    overlap_top30 = set(frame.head(30)["feature"]) & set(corr_frame.head(30)["feature"])
    print(f"Top-30 importance 与 Top-30 |corr| 交集: {len(overlap_top30)} 个特征")
    return frame


def analyze_train_test_shift(
    root: Path,
    output_dir: Path,
    figures_dir: Path,
    *,
    sample_rows: int,
) -> pd.DataFrame:
    feature_cols = MARKET_FIELDS + DERIVED_FEATURES
    train = load_parquet_frame(root, "train.parquet", sample_rows=sample_rows, include_label=False, selected_columns=feature_cols)
    test = load_parquet_frame(root, "test.parquet", sample_rows=sample_rows, include_label=False, selected_columns=feature_cols)
    train = add_basic_market_features(train)
    test = add_basic_market_features(test)
    all_features = MARKET_FIELDS + DERIVED_FEATURES

    rows = []
    for feature in all_features:
        train_vals = train[feature].astype(np.float64)
        test_vals = test[feature].astype(np.float64)
        train_std = float(train_vals.std())
        test_std = float(test_vals.std())
        rows.append(
            {
                "feature": feature,
                "feature_group": feature_group(feature),
                "train_mean": float(train_vals.mean()),
                "test_mean": float(test_vals.mean()),
                "train_std": train_std,
                "test_std": test_std,
                "mean_diff": float(test_vals.mean() - train_vals.mean()),
                "std_ratio_test_over_train": float(test_std / train_std) if train_std > 0 else np.nan,
            }
        )
    frame = pd.DataFrame(rows).sort_values("mean_diff", key=np.abs, ascending=False)
    frame.to_csv(output_dir / "train_test_shift_market.csv", index=False)

    anonymous = get_anonymous_feature_names(get_feature_columns(get_parquet_columns(raw_path(root, "train.parquet"))))
    sample_x = anonymous[:50]
    train_x = load_parquet_frame(root, "train.parquet", sample_rows=sample_rows, include_label=False, selected_columns=sample_x)
    test_x = load_parquet_frame(root, "test.parquet", sample_rows=sample_rows, include_label=False, selected_columns=sample_x)

    x_rows = []
    for feature in sample_x:
        train_vals = train_x[feature].astype(np.float64)
        test_vals = test_x[feature].astype(np.float64)
        train_std = float(train_vals.std())
        x_rows.append(
            {
                "feature": feature,
                "train_mean": float(train_vals.mean()),
                "test_mean": float(test_vals.mean()),
                "mean_diff": float(test_vals.mean() - train_vals.mean()),
                "std_ratio_test_over_train": float(test_vals.std() / train_std) if train_std > 0 else np.nan,
            }
        )
    pd.DataFrame(x_rows).to_csv(output_dir / "train_test_shift_x_sample50.csv", index=False)

    plt.figure(figsize=(9, 4))
    sns.barplot(data=frame, x="feature", y="mean_diff")
    plt.title("market/derived 特征 train-test 均值差")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "train_test_shift_market.png", dpi=150)
    plt.close()

    return frame


def build_recommendations(summary: dict) -> list[str]:
    recs: list[str] = []
    cluster_count = summary.get("x_cluster", {}).get("cluster_count", 0)
    if cluster_count and cluster_count < 40:
        recs.append(
            f"匿名特征在 |corr|>=0.6 下约 {cluster_count} 个簇，medoid 硬删特征风险高，优先全特征 + 强正则。"
        )
    holdout_ratio = summary.get("label", {}).get("holdout_std_over_train_std")
    if holdout_ratio and abs(holdout_ratio - 1.0) > 0.05:
        recs.append(
            f"Holdout 段 label 波动与训练段比值约 {holdout_ratio:.3f}，后续调参应结合 Purged CV，不宜只信 Holdout。"
        )
    recs.append("公开市场/派生特征单独特征相关性通常弱于 anonymous X，新特征应优先基于 Top importance X 或 market 时序。")
    recs.append("后续优化顺序建议: 调参 LGBM → CatBoost/XGB 异质集成 → 时序扩展特征；避免再跑 medoid 硬删。")
    return recs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 特征与 label 分析（官方题）")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--sample-rows", type=int, default=300_000, help="相关性/共线分析采样行数")
    parser.add_argument("--shift-sample-rows", type=int, default=100_000, help="train/test 漂移分析采样")
    parser.add_argument("--corr-threshold", type=float, default=0.6, help="X 特征共线聚类阈值")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "eda")
    figures_dir = ensure_dir(root / "outputs" / "figures" / "eda")
    setup_style()

    print("模块 A: label 分布与 Holdout 对比（全量）...")
    label_data = load_parquet_frame(root, "train.parquet", include_label=True, selected_columns=[TARGET_COL])
    label_summary = analyze_label(label_data, output_dir, figures_dir)

    print(f"模块 B: 特征-label 相关性（前 {args.sample_rows} 行）...")
    train_sample = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    train_sample = add_basic_market_features(train_sample)
    feature_cols = get_feature_columns(train_sample.columns)
    corr_frame = analyze_feature_label_correlation(
        train_sample,
        feature_cols,
        output_dir,
        figures_dir,
        sample_rows=None,
    )

    print(f"模块 C: 匿名特征共线簇（前 {args.sample_rows} 行, threshold={args.corr_threshold}）...")
    cluster_summary = analyze_x_clusters(
        train_sample,
        output_dir,
        figures_dir,
        sample_rows=args.sample_rows,
        threshold=args.corr_threshold,
    )

    print("模块 D: 调参 LGBM importance...")
    analyze_lgbm_importance(root, corr_frame, output_dir, figures_dir)

    print(f"模块 G: train/test 漂移（market 全量 + X 前 50，各 {args.shift_sample_rows} 行）...")
    analyze_train_test_shift(root, output_dir, figures_dir, sample_rows=args.shift_sample_rows)

    top_corr = corr_frame.head(10)[["feature", "abs_pearson", "feature_group"]].to_dict(orient="records")
    group_corr = pd.read_csv(output_dir / "feature_label_corr_by_group.csv").to_dict(orient="records")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_rows_corr": args.sample_rows,
        "sample_rows_shift": args.shift_sample_rows,
        "corr_threshold": args.corr_threshold,
        "label": label_summary,
        "x_cluster": cluster_summary,
        "top10_label_corr": top_corr,
        "label_corr_by_group": group_corr,
        "recommendations": [],
    }
    summary["recommendations"] = build_recommendations(summary)
    save_json(summary, output_dir / "eda_summary.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"EDA 结果目录: {output_dir}")
    print(f"EDA 图表目录: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
