from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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
    load_parquet_frame,
    save_json,
    validate_no_missing_or_infinite,
)
from feature_effectiveness.clustering import (
    filter_low_signal_features,
    get_anonymous_feature_names,
    select_medoid_features,
)
from feature_effectiveness.clustering import compute_feature_target_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 medoid 聚类阈值，目标 40-70 个代表特征")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--cluster-sample-rows", type=int, default=300_000)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7")
    parser.add_argument("--target-min", type=int, default=40)
    parser.add_argument("--target-max", type=int, default=70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    data = add_basic_market_features(data)
    anonymous = get_anonymous_feature_names(get_feature_columns(data.columns))
    validate_no_missing_or_infinite(data, anonymous + [TARGET_COL], context="medoid 扫描")

    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    scan_rows: list[dict[str, object]] = []
    chosen_threshold: float | None = None
    chosen_medoids: list[str] = []

    for threshold in thresholds:
        medoids, cluster_rows = select_medoid_features(
            data,
            anonymous,
            threshold=threshold,
            sample_rows=args.cluster_sample_rows,
        )
        scan_rows.append(
            {
                "corr_threshold": threshold,
                "cluster_count": len(set(row["cluster_id"] for row in cluster_rows)),
                "medoid_count": len(medoids),
            }
        )
        if args.target_min <= len(medoids) <= args.target_max:
            chosen_threshold = threshold
            chosen_medoids = medoids
            pd.DataFrame(cluster_rows).to_csv(
                output_dir / f"feature_clusters_{str(threshold).replace('.', '')}.csv",
                index=False,
            )

    if chosen_threshold is None:
        scan_frame = pd.DataFrame(scan_rows)
        best_row = scan_frame.iloc[(scan_frame["medoid_count"] - 55).abs().argsort().iloc[0]]
        chosen_threshold = float(best_row["corr_threshold"])
        chosen_medoids, cluster_rows = select_medoid_features(
            data,
            anonymous,
            threshold=chosen_threshold,
            sample_rows=args.cluster_sample_rows,
        )
        pd.DataFrame(cluster_rows).to_csv(output_dir / "feature_clusters_medoid.csv", index=False)

    target_corr = compute_feature_target_correlation(data, chosen_medoids, TARGET_COL)
    filtered_x, dropped_x = filter_low_signal_features(target_corr, threshold=1e-4)
    feature_cols = sorted(set(filtered_x + MARKET_FIELDS + DERIVED_FEATURES))

    payload = {
        "corr_threshold": chosen_threshold,
        "cluster_sample_rows": args.cluster_sample_rows,
        "medoid_features": chosen_medoids,
        "medoid_count": len(chosen_medoids),
        "filtered_x_count": len(filtered_x),
        "dropped_x_count": len(dropped_x),
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(payload, output_dir / "medoid_features_tuned.json")
    pd.DataFrame(scan_rows).to_csv(output_dir / "medoid_threshold_scan.csv", index=False)
    target_corr.sort_values(key=lambda series: series.abs(), ascending=False).to_frame(
        "target_corr"
    ).to_csv(output_dir / "medoid_target_corr.csv")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"已保存: {output_dir / 'medoid_features_tuned.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
