from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import add_basic_market_features
from data_preprocessing.preprocess import (  # noqa: E402
    DEFAULT_ROOT,
    TARGET_COL,
    ensure_dir,
    get_feature_columns,
    load_parquet_frame,
    save_json,
    validate_no_missing_or_infinite,
)
from prediction_task.metrics import evaluate_regression
from prediction_task.splits import time_order_split
from prediction_task.train_baseline import update_model_compare


def lgb_pearson_eval(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    labels = dataset.get_label()
    metrics = evaluate_regression(labels, preds)
    return "pearson", metrics["pearson"], True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练官方预测任务 LightGBM 初步主模型")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--sample-rows", type=int, default=None, help="只读取前 N 行做小样本验证")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--gap-rows", type=int, default=0, help="训练集和验证集之间的 gap 行数")
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


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    model_dir = ensure_dir(root / "models")

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    data = add_basic_market_features(data)
    feature_cols = get_feature_columns(data.columns)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="LightGBM 训练数据")

    train_idx, valid_idx = time_order_split(
        len(data),
        valid_fraction=args.valid_fraction,
        gap_rows=args.gap_rows,
    )
    X_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    X_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)

    params = {
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
    metrics = evaluate_regression(y_valid, valid_pred)

    model_path = model_dir / "official_lgbm.txt"
    feature_path = model_dir / "official_lgbm_features.json"
    model.save_model(model_path)
    save_json({"feature_columns": feature_cols}, feature_path)

    result = {
        "model": "lightgbm",
        "sample_rows": args.sample_rows or len(data),
        "train_rows": len(train_idx),
        "valid_rows": len(valid_idx),
        "valid_fraction": args.valid_fraction,
        "gap_rows": args.gap_rows,
        "metric_for_early_stop": args.metric_for_early_stop,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "max_depth": args.max_depth,
        "min_gain_to_split": args.min_gain_to_split,
        "best_iteration": int(model.best_iteration or args.num_boost_round),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }
    pd.DataFrame([result]).to_csv(output_dir / "official_lgbm_results.csv", index=False)
    pd.DataFrame(
        {
            "row_index": valid_idx,
            "y_true": y_valid,
            "y_pred": valid_pred,
            "model": "lightgbm",
        }
    ).to_csv(output_dir / "official_lgbm_valid_predictions.csv", index=False)
    save_json({"params": params}, output_dir / "official_lgbm_params.json")
    update_model_compare(output_dir)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"模型已保存: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
