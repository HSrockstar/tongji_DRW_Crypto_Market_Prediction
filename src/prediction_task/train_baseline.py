from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import add_basic_market_features
from data_preprocessing.preprocess import (  # noqa: E402
    DEFAULT_ROOT,
    TARGET_COL,
    ensure_dir,
    get_feature_columns,
    load_json,
    load_parquet_frame,
    save_json,
    validate_no_missing_or_infinite,
)
from prediction_task.metrics import evaluate_regression
from prediction_task.splits import time_order_split


def update_model_compare(output_dir: Path) -> None:
    compare_columns = [
        "model",
        "sample_rows",
        "train_rows",
        "valid_rows",
        "valid_fraction",
        "gap_rows",
        "pearson",
        "rmse",
        "mae",
        "generated_at",
    ]
    rows = []
    for path in [
        output_dir / "official_baseline_results.csv",
        output_dir / "official_lgbm_results.csv",
        output_dir / "official_lasso_results.csv",
    ]:
        if path.is_file():
            result = pd.read_csv(path)
            rows.append(result[[column for column in compare_columns if column in result.columns]])
    if rows:
        pd.concat(rows, ignore_index=True)[compare_columns].to_csv(
            output_dir / "official_model_compare.csv",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练官方预测任务 Ridge baseline")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--sample-rows", type=int, default=None, help="只读取前 N 行做小样本验证")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--gap-rows", type=int, default=0, help="训练集和验证集之间的 gap 行数")
    parser.add_argument("--feature-file", default=None, help="特征列 JSON，默认使用全部特征")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge 正则强度")
    parser.add_argument("--solver", default="lsqr", help="Ridge 求解器，全量数据默认使用 lsqr 避免 SVD 内存压力")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    model_dir = ensure_dir(root / "models")

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    data = add_basic_market_features(data)
    if args.feature_file:
        feature_path = Path(args.feature_file)
        if not feature_path.is_absolute():
            feature_path = root / feature_path
        feature_cols = load_json(feature_path)["feature_columns"]
    else:
        feature_cols = get_feature_columns(data.columns)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="Ridge 训练数据")

    train_idx, valid_idx = time_order_split(
        len(data),
        valid_fraction=args.valid_fraction,
        gap_rows=args.gap_rows,
    )
    X_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    X_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.alpha, solver=args.solver)),
        ]
    )
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)
    metrics = evaluate_regression(y_valid, valid_pred)

    model_path = model_dir / ("official_ridge_selected.pkl" if args.feature_file else "official_ridge.pkl")
    feature_path = model_dir / (
        "official_ridge_selected_features.json" if args.feature_file else "official_ridge_features.json"
    )
    if args.feature_file:
        model_path = output_dir / "selected_ridge.pkl"
        feature_path = output_dir / "selected_ridge_features.json"
    joblib.dump(model, model_path)
    save_json({"feature_columns": feature_cols}, feature_path)

    result = {
        "model": "ridge",
        "sample_rows": args.sample_rows or len(data),
        "train_rows": len(train_idx),
        "valid_rows": len(valid_idx),
        "valid_fraction": args.valid_fraction,
        "gap_rows": args.gap_rows,
        "alpha": args.alpha,
        "solver": args.solver,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **metrics,
    }
    pd.DataFrame([result]).to_csv(output_dir / "official_baseline_results.csv", index=False)
    pd.DataFrame(
        {
            "row_index": valid_idx,
            "y_true": y_valid,
            "y_pred": valid_pred,
            "model": "ridge",
        }
    ).to_csv(output_dir / "official_baseline_valid_predictions.csv", index=False)
    update_model_compare(output_dir)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"模型已保存: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
