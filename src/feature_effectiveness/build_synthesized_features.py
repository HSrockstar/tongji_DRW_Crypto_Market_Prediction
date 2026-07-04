from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import DERIVED_FEATURES, MARKET_FIELDS, add_basic_market_features
from data_preprocessing.preprocess import DEFAULT_ROOT, TARGET_COL, ensure_dir, load_json, load_parquet_frame, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冠军线简化：Top 特征线性组合")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--importance-file", default="outputs/experiments/feature_importance_summary.csv")
    parser.add_argument("--base-feature-file", default="outputs/experiments/medoid_features_tuned.json")
    parser.add_argument("--top-k", type=int, default=10, help="参与组合的重要 X 特征数")
    parser.add_argument("--max-combos", type=int, default=30, help="最多保留组合特征数")
    parser.add_argument("--corr-threshold", type=float, default=0.005, help="组合与 label 最小 |corr|")
    return parser.parse_args()


def build_combo_candidates(
    data: pd.DataFrame,
    features: list[str],
    *,
    max_combos: int,
    corr_threshold: float,
) -> tuple[list[dict[str, str]], pd.DataFrame]:
    y = data[TARGET_COL].astype(np.float64)
    combo_defs: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []
    eps = 1e-6

    for left, right in combinations(features, 2):
        a = data[left].astype(np.float64)
        b = data[right].astype(np.float64)
        candidates = [
            (f"syn_add_{left}_{right}", "add", a + b),
            (f"syn_sub_{left}_{right}", "sub", a - b),
            (f"syn_div_{left}_{right}", "div", a / (np.abs(b) + eps)),
            (f"syn_mul_{left}_{right}", "mul", a * b),
        ]
        for name, op, values in candidates:
            if values.std() == 0 or y.std() == 0:
                corr = 0.0
            else:
                corr = float(values.corr(y, method="pearson"))
            rows.append(
                {
                    "name": name,
                    "left": left,
                    "right": right,
                    "op": op,
                    "target_corr": corr,
                }
            )
            if abs(corr) >= corr_threshold:
                combo_defs.append({"name": name, "left": left, "right": right, "op": op})

    candidate_frame = pd.DataFrame(rows).sort_values("target_corr", key=lambda s: s.abs(), ascending=False)
    selected = candidate_frame.head(max_combos)
    combo_defs = [
        {"name": row["name"], "left": row["left"], "right": row["right"], "op": row["op"]}
        for _, row in selected.iterrows()
    ]
    return combo_defs, candidate_frame


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    data = add_basic_market_features(data)

    importance_path = root / args.importance_file
    importance = pd.read_csv(importance_path)
    top_x = (
        importance[importance["feature"].str.startswith("X")]
        .sort_values(["fold_count", "mean_importance"], ascending=[False, False])
        .head(args.top_k)["feature"]
        .tolist()
    )

    base_path = root / args.base_feature_file
    base_payload = load_json(base_path)
    base_features = base_payload["feature_columns"]

    combo_defs, candidate_frame = build_combo_candidates(
        data,
        top_x,
        max_combos=args.max_combos,
        corr_threshold=args.corr_threshold,
    )
    combo_names = [combo["name"] for combo in combo_defs]
    feature_columns = sorted(set(base_features + combo_names))

    payload = {
        "top_x_features": top_x,
        "combo_count": len(combo_defs),
        "combo_defs": combo_defs,
        "base_feature_file": str(args.base_feature_file),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(payload, output_dir / "synthesized_features.json")
    candidate_frame.to_csv(output_dir / "synthesized_feature_candidates.csv", index=False)
    print(json.dumps({k: payload[k] for k in payload if k != "combo_defs"}, ensure_ascii=False, indent=2))
    print(f"已保存: {output_dir / 'synthesized_features.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
