from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import add_basic_market_features, add_synthesized_features
from data_preprocessing.preprocess import DEFAULT_ROOT, TARGET_COL, ensure_dir, load_json, load_parquet_frame, save_json
from prediction_task.metrics import evaluate_regression
from prediction_task.splits import time_order_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段2：组合特征 holdout 评估 + 更新特征文件")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--synthesized-file", default="outputs/experiments/synthesized_features.json")
    parser.add_argument("--lgbm-params-file", default="outputs/experiments/lgbm_best_params.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")

    payload = load_json(root / args.synthesized_file)
    combo_defs = payload["combo_defs"]
    base_features = payload["feature_columns"]

    data = load_parquet_frame(root, "train.parquet", include_label=True)
    data = add_basic_market_features(data)
    data = add_synthesized_features(data, combo_defs)

    lgbm_params_path = root / args.lgbm_params_file
    if lgbm_params_path.is_file():
        lgbm_payload = load_json(lgbm_params_path)
        params = lgbm_payload["params"]
    else:
        params = {
            "num_leaves": 15,
            "min_data_in_leaf": 500,
            "lambda_l1": 0.0,
            "lambda_l2": 50.0,
            "feature_fraction": 0.7,
        }

    cmd = [
        sys.executable,
        str(root / "src" / "prediction_task" / "train_lgbm.py"),
        "--root",
        str(root),
        "--feature-file",
        str((output_dir / "synthesized_lgbm_features.json").relative_to(root)),
        "--num-leaves",
        str(int(params["num_leaves"])),
        "--min-data-in-leaf",
        str(int(params["min_data_in_leaf"])),
        "--lambda-l2",
        str(params["lambda_l2"]),
        "--feature-fraction",
        str(params["feature_fraction"]),
    ]
    save_json(
        {
            "feature_columns": base_features,
            "combo_defs": combo_defs,
        },
        output_dir / "synthesized_lgbm_features.json",
    )
    print("训练组合特征 LightGBM:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    results_path = output_dir / "synthesized_lgbm_results.csv"
    if results_path.is_file():
        import pandas as pd

        pearson = float(pd.read_csv(results_path).iloc[0]["pearson"])
        save_json(
            {
                "feature_file": str(output_dir / "synthesized_lgbm_features.json"),
                "combo_count": len(combo_defs),
                "holdout_pearson": pearson,
            },
            output_dir / "phase2_combo_summary.json",
        )
        print(f"组合特征 LGBM holdout pearson={pearson:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
