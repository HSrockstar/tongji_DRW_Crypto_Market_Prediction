from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
from feature_effectiveness.clustering import get_anonymous_feature_names
from prediction_task.metrics import evaluate_regression
from prediction_task.run_time_cv_experiments import fit_predict_lgbm
from prediction_task.splits import purged_group_time_series_splits, time_order_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightGBM purged CV 网格调参")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--feature-set", choices=["full", "anonymous_x", "both"], default="both")
    parser.add_argument("--feature-file", default=None, help="自定义特征 JSON，覆盖 feature-set")
    parser.add_argument("--include-medoid", action="store_true", help="额外加入 medoid_features_tuned.json")
    parser.add_argument("--n-groups", type=int, default=6)
    parser.add_argument("--gap-groups", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="缩小搜索空间")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--metric-for-early-stop", choices=["pearson", "rmse"], default="pearson")
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-gain-to-split", type=float, default=0.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-solver", default="lsqr")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--log-period", type=int, default=100)
    parser.add_argument("--retrain-best", action="store_true", help="调参完成后用最优参数重训 holdout 模型")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="快速模式：5 组预设参数 + holdout 评估（约 30 分钟）",
    )
    return parser.parse_args()


FAST_PARAM_PRESETS: list[dict[str, float | int]] = [
    {"num_leaves": 15, "min_data_in_leaf": 500, "lambda_l1": 0.0, "lambda_l2": 50.0, "feature_fraction": 0.7},
    {"num_leaves": 31, "min_data_in_leaf": 500, "lambda_l1": 0.0, "lambda_l2": 50.0, "feature_fraction": 0.7},
    {"num_leaves": 15, "min_data_in_leaf": 1000, "lambda_l1": 0.0, "lambda_l2": 100.0, "feature_fraction": 0.5},
    {"num_leaves": 31, "min_data_in_leaf": 300, "lambda_l1": 0.0, "lambda_l2": 20.0, "feature_fraction": 0.9},
    {"num_leaves": 63, "min_data_in_leaf": 500, "lambda_l1": 0.0, "lambda_l2": 50.0, "feature_fraction": 0.7},
]


def build_feature_sets(data: pd.DataFrame, args: argparse.Namespace) -> dict[str, list[str]]:
    if args.feature_file:
        feature_path = Path(args.feature_file)
        if not feature_path.is_absolute():
            feature_path = Path(args.root).expanduser().resolve() / feature_path
        return {"custom": load_json(feature_path)["feature_columns"]}

    all_features = get_feature_columns(data.columns)
    anonymous = get_anonymous_feature_names(all_features)
    sets: dict[str, list[str]] = {}
    if args.feature_set in {"full", "both"}:
        sets["full"] = all_features
    if args.feature_set in {"anonymous_x", "both"}:
        sets["anonymous_x"] = anonymous
    if args.include_medoid:
        medoid_path = Path(args.root).expanduser().resolve() / "outputs" / "experiments" / "medoid_features_tuned.json"
        if medoid_path.is_file():
            sets["medoid_tuned"] = load_json(medoid_path)["feature_columns"]
    return sets


def build_param_grid(quick: bool) -> list[dict[str, float | int]]:
    if quick:
        grid = {
            "num_leaves": [15, 31],
            "min_data_in_leaf": [300, 500],
            "lambda_l1": [0.0],
            "lambda_l2": [10.0, 50.0],
            "feature_fraction": [0.7, 0.9],
        }
    else:
        grid = {
            "num_leaves": [15, 31, 63],
            "min_data_in_leaf": [300, 500, 1000],
            "lambda_l1": [0.0, 10.0],
            "lambda_l2": [10.0, 50.0, 100.0],
            "feature_fraction": [0.5, 0.7, 0.9],
        }
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[key] for key in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def make_args_namespace(base: argparse.Namespace, params: dict[str, float | int]) -> argparse.Namespace:
    merged = vars(base).copy()
    merged.update(params)
    merged.setdefault("num_threads", 0)
    merged.setdefault("log_period", 100)
    return argparse.Namespace(**merged)


def run_purged_cv(
    data: pd.DataFrame,
    feature_cols: list[str],
    splits: list[dict[str, object]],
    args: argparse.Namespace,
    params: dict[str, float | int],
) -> dict[str, object]:
    fold_args = make_args_namespace(args, params)
    pearsons: list[float] = []
    for split in splits:
        valid_idx = split["valid_idx"]
        y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
        valid_pred, _ = fit_predict_lgbm(data, split, feature_cols, fold_args)
        metrics = evaluate_regression(y_valid, valid_pred)
        pearsons.append(metrics["pearson"])
    pearson_array = np.asarray(pearsons, dtype=np.float64)
    return {
        **params,
        "pearson_mean": float(pearson_array.mean()),
        "pearson_std": float(pearson_array.std()),
        "fold_pearsons": pearsons,
    }


def run_holdout_eval(
    data: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    params: dict[str, float | int],
) -> dict[str, object]:
    train_idx, valid_idx = time_order_split(len(data), valid_fraction=0.2, gap_rows=0)
    split = {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
    }
    fold_args = make_args_namespace(args, params)
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
    valid_pred, model_info = fit_predict_lgbm(data, split, feature_cols, fold_args)
    metrics = evaluate_regression(y_valid, valid_pred)
    return {**params, **metrics, **model_info}


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    data = add_basic_market_features(data)
    generated_at = datetime.now().isoformat(timespec="seconds")

    if args.fast:
        feature_cols = get_feature_columns(data.columns)
        validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="LGBM fast tune")
        param_grid = FAST_PARAM_PRESETS
        results: list[dict[str, object]] = []
        best_row: dict[str, object] | None = None
        print(f"快速调参: full {len(feature_cols)} 特征, {len(param_grid)} 组, holdout 评估")
        for idx, params in enumerate(param_grid, start=1):
            summary = run_holdout_eval(data, feature_cols, args, params)
            row = {
                "feature_set": "full",
                "feature_count": len(feature_cols),
                "eval_method": "holdout",
                "generated_at": generated_at,
                **summary,
            }
            results.append(row)
            if best_row is None or row["pearson"] > best_row["pearson"]:
                best_row = row
            print(f"  [{idx}/{len(param_grid)}] pearson={summary['pearson']:.6f} params={params}")
        assert best_row is not None
        pd.DataFrame(results).sort_values("pearson", ascending=False).to_csv(
            output_dir / "lgbm_tune_results.csv", index=False
        )
        best_payload = {
            "feature_set": "full",
            "feature_count": len(feature_cols),
            "pearson_holdout": best_row["pearson"],
            "eval_method": "holdout",
            "params": {key: best_row[key] for key in param_grid[0]},
            "feature_columns": feature_cols,
            "generated_at": generated_at,
        }
        save_json(best_payload, output_dir / "lgbm_best_params.json")
        print(f"最优 holdout pearson={best_row['pearson']:.6f}")
        if args.retrain_best:
            cmd = [
                sys.executable,
                str(root / "src" / "prediction_task" / "train_lgbm.py"),
                "--root",
                str(root),
                "--num-leaves",
                str(int(best_payload["params"]["num_leaves"])),
                "--min-data-in-leaf",
                str(int(best_payload["params"]["min_data_in_leaf"])),
                "--lambda-l2",
                str(best_payload["params"]["lambda_l2"]),
                "--feature-fraction",
                str(best_payload["params"]["feature_fraction"]),
            ]
            print("重训最优 LightGBM:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        return 0

    feature_sets = build_feature_sets(data, args)
    splits = purged_group_time_series_splits(len(data), n_groups=args.n_groups, gap_groups=args.gap_groups)
    param_grid = build_param_grid(args.quick)
    generated_at = datetime.now().isoformat(timespec="seconds")

    results: list[dict[str, object]] = []
    best_by_set: dict[str, dict[str, object]] = {}
    for set_name, feature_cols in feature_sets.items():
        validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context=f"LGBM tune {set_name}")
        print(f"调参特征集 {set_name}: {len(feature_cols)} 特征, {len(param_grid)} 组参数, {len(splits)} 折")
        best_row: dict[str, object] | None = None
        for idx, params in enumerate(param_grid, start=1):
            summary = run_purged_cv(data, feature_cols, splits, args, params)
            row = {
                "feature_set": set_name,
                "feature_count": len(feature_cols),
                "generated_at": generated_at,
                **summary,
            }
            results.append(row)
            if best_row is None or row["pearson_mean"] > best_row["pearson_mean"]:
                best_row = row
            if idx % 10 == 0 or idx == len(param_grid):
                print(
                    f"  [{set_name}] {idx}/{len(param_grid)} "
                    f"current={params} pearson_mean={summary['pearson_mean']:.6f}"
                )
        if best_row is not None:
            best_by_set[set_name] = best_row
            print(
                f"最优 [{set_name}] pearson_mean={best_row['pearson_mean']:.6f} "
                f"std={best_row['pearson_std']:.6f} params={{{', '.join(f'{k}={best_row[k]}' for k in param_grid[0])}}}"
            )

    result_frame = pd.DataFrame(results).sort_values(["feature_set", "pearson_mean"], ascending=[True, False])
    result_frame.drop(columns=["fold_pearsons"], errors="ignore").to_csv(output_dir / "lgbm_tune_results.csv", index=False)

    overall_best_name = max(best_by_set, key=lambda name: float(best_by_set[name]["pearson_mean"]))
    overall_best = best_by_set[overall_best_name]
    best_payload = {
        "feature_set": overall_best_name,
        "feature_count": overall_best["feature_count"],
        "pearson_mean": overall_best["pearson_mean"],
        "pearson_std": overall_best["pearson_std"],
        "params": {key: overall_best[key] for key in param_grid[0]},
        "feature_columns": feature_sets[overall_best_name],
        "generated_at": generated_at,
    }
    if args.feature_file:
        best_payload["feature_file"] = str(args.feature_file)
    save_json(best_payload, output_dir / "lgbm_best_params.json")
    save_json({name: row for name, row in best_by_set.items()}, output_dir / "lgbm_best_by_feature_set.json")
    print(f"全局最优: {overall_best_name}, pearson_mean={overall_best['pearson_mean']:.6f}")
    print(f"结果已保存: {output_dir / 'lgbm_tune_results.csv'}")

    if args.retrain_best:
        feature_file = output_dir / f"lgbm_best_features_{overall_best_name}.json"
        save_json({"feature_columns": feature_sets[overall_best_name]}, feature_file)
        cmd = [
            sys.executable,
            str(root / "src" / "prediction_task" / "train_lgbm.py"),
            "--root",
            str(root),
            "--num-leaves",
            str(int(best_payload["params"]["num_leaves"])),
            "--min-data-in-leaf",
            str(int(best_payload["params"]["min_data_in_leaf"])),
            "--lambda-l1",
            str(best_payload["params"]["lambda_l1"]),
            "--lambda-l2",
            str(best_payload["params"]["lambda_l2"]),
            "--feature-fraction",
            str(best_payload["params"]["feature_fraction"]),
        ]
        if overall_best_name != "full":
            cmd.extend(["--feature-file", str(feature_file.relative_to(root))])
        print("重训最优 LightGBM:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        save_json(best_payload, output_dir / "lgbm_best_holdout_model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
