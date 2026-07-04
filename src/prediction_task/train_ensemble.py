from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import add_basic_market_features
from data_preprocessing.preprocess import (
    DEFAULT_ROOT,
    TARGET_COL,
    ensure_dir,
    get_feature_columns,
    load_json,
    load_parquet_frame,
    raw_path,
    save_json,
    validate_no_missing_or_infinite,
)
from prediction_task.make_submission import load_lgbm_model
from prediction_task.metrics import evaluate_regression
from prediction_task.run_time_cv_experiments import fit_predict_lgbm, fit_predict_ridge
from prediction_task.splits import purged_group_time_series_splits, time_order_split
from prediction_task.train_lgbm import save_lgbm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge + LightGBM 加权集成")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--lgbm-params-file", default="outputs/experiments/lgbm_best_params.json")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-solver", default="lsqr")
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--gap-rows", type=int, default=0)
    parser.add_argument("--n-groups", type=int, default=6)
    parser.add_argument("--gap-groups", type=int, default=1)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--metric-for-early-stop", choices=["pearson", "rmse"], default="pearson")
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-gain-to-split", type=float, default=0.0)
    parser.add_argument("--ridge-solver-cv", default="lsqr")
    parser.add_argument(
        "--weight-search",
        choices=["holdout", "purged"],
        default="holdout",
        help="集成权重搜索方式，holdout 更快",
    )
    return parser.parse_args()


def default_lgbm_config(feature_cols: list[str]) -> tuple[list[str], argparse.Namespace]:
    lgbm_args = argparse.Namespace(
        ridge_alpha=1.0,
        ridge_solver="lsqr",
        learning_rate=0.01,
        num_leaves=15,
        min_data_in_leaf=300,
        lambda_l1=0.0,
        lambda_l2=20.0,
        feature_fraction=0.9,
        num_boost_round=3000,
        early_stopping_rounds=200,
        metric_for_early_stop="pearson",
        bagging_fraction=0.9,
        max_depth=-1,
        min_gain_to_split=0.0,
        num_threads=0,
        log_period=100,
    )
    return feature_cols, lgbm_args


def load_lgbm_config(root: Path, params_file: str, fallback_features: list[str]) -> tuple[list[str], argparse.Namespace]:
    path = Path(params_file)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        print(f"未找到 {path}，使用默认 LightGBM 参数")
        return default_lgbm_config(fallback_features)
    payload = load_json(path)
    feature_cols = payload["feature_columns"]
    params = payload["params"]
    lgbm_args = argparse.Namespace(
        ridge_alpha=1.0,
        ridge_solver="lsqr",
        learning_rate=0.01,
        num_leaves=int(params["num_leaves"]),
        min_data_in_leaf=int(params["min_data_in_leaf"]),
        lambda_l1=float(params["lambda_l1"]),
        lambda_l2=float(params["lambda_l2"]),
        feature_fraction=float(params["feature_fraction"]),
        num_boost_round=3000,
        early_stopping_rounds=200,
        metric_for_early_stop="pearson",
        bagging_fraction=0.9,
        max_depth=-1,
        min_gain_to_split=0.0,
        num_threads=0,
        log_period=100,
    )
    return feature_cols, lgbm_args


def search_blend_weight_holdout(
    y_valid: np.ndarray,
    ridge_pred: np.ndarray,
    lgbm_pred: np.ndarray,
    weight_step: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ridge_weight in np.arange(0.5, 0.951, weight_step):
        lgbm_weight = 1.0 - ridge_weight
        blend = ridge_weight * ridge_pred + lgbm_weight * lgbm_pred
        metrics = evaluate_regression(y_valid, blend)
        rows.append(
            {
                "ridge_weight": float(ridge_weight),
                "lgbm_weight": float(lgbm_weight),
                "pearson_mean": metrics["pearson"],
                "pearson_std": 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_mean", ascending=False)


def search_blend_weight(
    data: pd.DataFrame,
    splits: list[dict[str, object]],
    ridge_features: list[str],
    lgbm_features: list[str],
    args: argparse.Namespace,
    lgbm_args: argparse.Namespace,
) -> pd.DataFrame:
    weights = np.arange(0.5, 0.951, args.weight_step)
    rows: list[dict[str, object]] = []
    ridge_cv_args = argparse.Namespace(ridge_alpha=args.ridge_alpha, ridge_solver=args.ridge_solver_cv)
    for ridge_weight in weights:
        lgbm_weight = 1.0 - ridge_weight
        fold_pearsons: list[float] = []
        for split in splits:
            valid_idx = split["valid_idx"]
            y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
            ridge_pred, _ = fit_predict_ridge(data, split, ridge_features, ridge_cv_args)
            lgbm_pred, _ = fit_predict_lgbm(data, split, lgbm_features, lgbm_args)
            blend = ridge_weight * ridge_pred + lgbm_weight * lgbm_pred
            fold_pearsons.append(evaluate_regression(y_valid, blend)["pearson"])
        pearson_array = np.asarray(fold_pearsons, dtype=np.float64)
        rows.append(
            {
                "ridge_weight": float(ridge_weight),
                "lgbm_weight": float(lgbm_weight),
                "pearson_mean": float(pearson_array.mean()),
                "pearson_std": float(pearson_array.std()),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_mean", ascending=False)


def lgb_pearson_eval(preds: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    labels = dataset.get_label()
    metrics = evaluate_regression(labels, preds)
    return "pearson", metrics["pearson"], True


def train_final_models(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    ridge_features: list[str],
    lgbm_features: list[str],
    args: argparse.Namespace,
    lgbm_args: argparse.Namespace,
) -> tuple[Pipeline, lgb.Booster]:
    ridge_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=args.ridge_alpha, solver=args.ridge_solver)),
        ]
    )
    ridge_model.fit(data.iloc[train_idx][ridge_features], data.iloc[train_idx][TARGET_COL])

    params = {
        "objective": "regression",
        "metric": "None",
        "learning_rate": lgbm_args.learning_rate,
        "num_leaves": lgbm_args.num_leaves,
        "max_depth": lgbm_args.max_depth,
        "min_data_in_leaf": lgbm_args.min_data_in_leaf,
        "feature_fraction": lgbm_args.feature_fraction,
        "bagging_fraction": lgbm_args.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l1": lgbm_args.lambda_l1,
        "lambda_l2": lgbm_args.lambda_l2,
        "min_gain_to_split": lgbm_args.min_gain_to_split,
        "seed": 42,
        "verbosity": -1,
    }
    X_train = data.iloc[train_idx][lgbm_features]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    X_valid = data.iloc[valid_idx][lgbm_features]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=lgbm_features)
    valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=lgbm_features, reference=train_set)
    lgbm_model = lgb.train(
        params,
        train_set,
        num_boost_round=lgbm_args.num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        feval=lgb_pearson_eval,
        callbacks=[
            lgb.early_stopping(lgbm_args.early_stopping_rounds),
            lgb.log_evaluation(100),
        ],
    )
    return ridge_model, lgbm_model


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    model_dir = ensure_dir(root / "models")
    submission_dir = ensure_dir(root / "outputs" / "submissions")

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    data = add_basic_market_features(data)
    ridge_features = get_feature_columns(data.columns)
    lgbm_features, lgbm_args = load_lgbm_config(root, args.lgbm_params_file, ridge_features)
    validate_no_missing_or_infinite(data, ridge_features + lgbm_features + [TARGET_COL], context="集成训练")

    train_idx, valid_idx = time_order_split(
        len(data),
        valid_fraction=args.valid_fraction,
        gap_rows=args.gap_rows,
    )
    ridge_model, lgbm_model = train_final_models(
        data, train_idx, valid_idx, ridge_features, lgbm_features, args, lgbm_args
    )
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
    ridge_pred = ridge_model.predict(data.iloc[valid_idx][ridge_features])
    lgbm_pred = lgbm_model.predict(data.iloc[valid_idx][lgbm_features], num_iteration=lgbm_model.best_iteration)

    if args.weight_search == "holdout":
        weight_frame = search_blend_weight_holdout(y_valid, ridge_pred, lgbm_pred, args.weight_step)
    else:
        splits = purged_group_time_series_splits(len(data), n_groups=args.n_groups, gap_groups=args.gap_groups)
        weight_frame = search_blend_weight(data, splits, ridge_features, lgbm_features, args, lgbm_args)
    weight_frame.to_csv(output_dir / "ensemble_weight_search.csv", index=False)
    best_weight = float(weight_frame.iloc[0]["ridge_weight"])
    print(weight_frame.head(5).to_string(index=False))
    print(f"最优 ridge_weight={best_weight:.2f}")

    blend_pred = best_weight * ridge_pred + (1.0 - best_weight) * lgbm_pred
    holdout_metrics = evaluate_regression(y_valid, blend_pred)
    ridge_metrics = evaluate_regression(y_valid, ridge_pred)
    lgbm_metrics = evaluate_regression(y_valid, lgbm_pred)

    ridge_path = model_dir / "ensemble_ridge.pkl"
    lgbm_path = output_dir / "ensemble_lgbm.txt"
    joblib.dump(ridge_model, ridge_path)
    save_lgbm_model(lgbm_model, lgbm_path)
    save_json({"feature_columns": ridge_features}, model_dir / "ensemble_ridge_features.json")
    save_json({"feature_columns": lgbm_features}, output_dir / "ensemble_lgbm_features.json")
    ensemble_meta = {
        "ridge_weight": best_weight,
        "lgbm_weight": 1.0 - best_weight,
        "ridge_features": len(ridge_features),
        "lgbm_features": len(lgbm_features),
        "lgbm_params_file": args.lgbm_params_file,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "holdout_blend": holdout_metrics,
        "holdout_ridge": ridge_metrics,
        "holdout_lgbm": lgbm_metrics,
    }
    save_json(ensemble_meta, output_dir / "ensemble_meta.json")

    test_data = load_parquet_frame(root, "test.parquet", sample_rows=args.sample_rows, include_label=False)
    test_data = add_basic_market_features(test_data)
    test_ridge = ridge_model.predict(test_data[ridge_features])
    test_lgbm = lgbm_model.predict(test_data[lgbm_features], num_iteration=lgbm_model.best_iteration)
    test_blend = best_weight * test_ridge + (1.0 - best_weight) * test_lgbm

    sample_submission = pd.read_csv(raw_path(root, "sample_submission.csv"))
    if args.sample_rows is not None:
        submission = sample_submission.head(args.sample_rows).copy()
        output_path = submission_dir / "smoke_submission_ensemble.csv"
    else:
        submission = sample_submission.copy()
        output_path = submission_dir / "submission_ensemble.csv"
    submission["prediction"] = np.asarray(test_blend, dtype=np.float64)
    submission.to_csv(output_path, index=False)

    summary = {
        **ensemble_meta,
        "submission": str(output_path),
        "holdout_pearson_blend": holdout_metrics["pearson"],
        "holdout_pearson_ridge": ridge_metrics["pearson"],
        "holdout_pearson_lgbm": lgbm_metrics["pearson"],
    }
    pd.DataFrame([summary]).to_csv(output_dir / "ensemble_holdout_results.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
