from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import MARKET_FIELDS, add_basic_market_features
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
from prediction_task.metrics import evaluate_regression
from prediction_task.run_time_cv_experiments import lgb_pearson_eval
from prediction_task.splits import time_order_split
from prediction_task.train_lgbm import save_lgbm_model

BASELINE_PEARSON = 0.10235862829968674
BASELINE_NAME = "phase1b_tuned_lgbm"
TOP_X_FOR_LAG = ["X466", "X33", "X752", "X272", "X758"]
MARKET_TEMPORAL_COLS = ["volume", "book_imbalance", "trade_imbalance"]


@dataclass
class ExperimentRecord:
    step: str
    experiment: str
    holdout_pearson: float
    holdout_rmse: float
    holdout_mae: float
    purged_cv_mean: float | None = None
    purged_cv_std: float | None = None
    vs_baseline: float = 0.0
    beats_baseline: bool = False
    submission_file: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def log_message(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_best_lgbm_params(root: Path) -> dict[str, float | int]:
    path = root / "outputs" / "experiments" / "lgbm_best_params.json"
    payload = load_json(path)
    return payload["params"]


def make_lgbm_args(
    root: Path,
    params: dict[str, float | int],
    seed: int = 42,
    *,
    smoke_test: bool = False,
) -> argparse.Namespace:
    merged = {
        "root": str(root),
        "sample_rows": None,
        "valid_fraction": 0.2,
        "gap_rows": 0,
        "feature_file": None,
        "learning_rate": 0.01,
        "num_boost_round": 20 if smoke_test else 3000,
        "early_stopping_rounds": 5 if smoke_test else 200,
        "metric_for_early_stop": "pearson",
        "bagging_fraction": 0.9,
        "max_depth": -1,
        "min_gain_to_split": 0.0,
        "num_threads": 0,
        "log_period": 10 if smoke_test else 200,
        "seed": seed,
        **params,
    }
    return argparse.Namespace(**merged)


def add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    result = add_basic_market_features(data)
    for column in MARKET_TEMPORAL_COLS:
        if column not in result.columns:
            continue
        series = result[column].astype(np.float32)
        for lag in (1, 5, 10):
            result[f"{column}_lag{lag}"] = series.shift(lag).fillna(0.0).astype(np.float32)
        result[f"{column}_rollmean5"] = series.rolling(5, min_periods=1).mean().astype(np.float32)
    for feature in TOP_X_FOR_LAG:
        if feature in result.columns:
            result[f"{feature}_lag1"] = result[feature].astype(np.float32).shift(1).fillna(0.0).astype(np.float32)
    return result


def prepare_dataset(root: Path, *, temporal: bool = False, sample_rows: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    data = load_parquet_frame(root, "train.parquet", sample_rows=sample_rows, include_label=True)
    data = add_temporal_features(data) if temporal else add_basic_market_features(data)
    feature_cols = get_feature_columns(data.columns)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="overnight optimization")
    return data, feature_cols


def train_lgbm_model(
    data: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    valid_idx: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[lgb.Booster, np.ndarray | None, dict[str, float]]:
    X_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    params = {
        "objective": "regression",
        "metric": "None",
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
        "seed": args.seed,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(args.log_period)]
    if valid_idx is not None and len(valid_idx) > 0:
        X_valid = data.iloc[valid_idx][feature_cols]
        y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
        valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_cols, reference=train_set)
        valid_sets = [train_set, valid_set]
        valid_names = ["train", "valid"]
        callbacks = [
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(args.log_period),
        ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=lgb_pearson_eval,
        callbacks=callbacks,
    )
    valid_pred = None
    metrics: dict[str, float] = {}
    if valid_idx is not None and len(valid_idx) > 0:
        valid_pred = model.predict(data.iloc[valid_idx][feature_cols], num_iteration=model.best_iteration)
        metrics = evaluate_regression(
            data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64),
            valid_pred,
        )
    return model, valid_pred, metrics


def train_catboost_holdout(
    data: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int = 42,
    *,
    smoke_test: bool = False,
) -> tuple[CatBoostRegressor, np.ndarray, dict[str, float]]:
    model = CatBoostRegressor(
        iterations=20 if smoke_test else 3000,
        learning_rate=0.01,
        depth=6,
        l2_leaf_reg=50.0,
        random_strength=1.0,
        rsm=0.7,
        min_data_in_leaf=500,
        early_stopping_rounds=5 if smoke_test else 200,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=10 if smoke_test else 200,
    )
    X_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL]
    X_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL]
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    valid_pred = model.predict(X_valid).astype(np.float64)
    metrics = evaluate_regression(y_valid.to_numpy(dtype=np.float64), valid_pred)
    return model, valid_pred, metrics


def train_xgb_holdout(
    data: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int = 42,
    *,
    smoke_test: bool = False,
) -> tuple[np.ndarray, dict[str, float], XGBRegressor]:
    model = XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=500,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=50.0,
        n_estimators=20 if smoke_test else 3000,
        early_stopping_rounds=5 if smoke_test else 200,
        random_state=seed,
        n_jobs=-1,
        verbosity=0 if smoke_test else 1,
    )
    X_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL]
    X_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL]
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False if smoke_test else 200)
    valid_pred = model.predict(X_valid).astype(np.float64)
    metrics = evaluate_regression(y_valid.to_numpy(dtype=np.float64), valid_pred)
    return valid_pred, metrics, model


def save_submission(root: Path, predictions: np.ndarray, file_name: str, submission_dir: Path | None = None) -> Path:
    submission_dir = ensure_dir(submission_dir or (root / "outputs" / "submissions"))
    template = pd.read_csv(raw_path(root, "sample_submission.csv"))
    if len(template) != len(predictions):
        template = template.head(len(predictions)).copy()
    output = template.copy()
    output["prediction"] = np.asarray(predictions, dtype=np.float64)
    output_path = submission_dir / file_name
    output.to_csv(output_path, index=False)
    return output_path


def display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def record_to_row(record: ExperimentRecord) -> dict[str, Any]:
    return {
        "step": record.step,
        "experiment": record.experiment,
        "holdout_pearson": record.holdout_pearson,
        "holdout_rmse": record.holdout_rmse,
        "holdout_mae": record.holdout_mae,
        "purged_cv_mean": record.purged_cv_mean,
        "purged_cv_std": record.purged_cv_std,
        "vs_baseline": record.vs_baseline,
        "beats_baseline": record.beats_baseline,
        "submission_file": record.submission_file,
        "notes": record.notes,
        **record.extra,
    }


def finalize_records(records: list[ExperimentRecord]) -> pd.DataFrame:
    baseline = ExperimentRecord(
        step="baseline",
        experiment=BASELINE_NAME,
        holdout_pearson=BASELINE_PEARSON,
        holdout_rmse=1.0391435114885461,
        holdout_mae=0.6984824662618114,
        vs_baseline=0.0,
        beats_baseline=False,
        submission_file="outputs/submissions/submission.csv",
        notes="保留不动的主结果",
    )
    all_records = [baseline, *records]
    frame = pd.DataFrame([record_to_row(item) for item in all_records])
    frame["rank_by_pearson"] = frame["holdout_pearson"].rank(ascending=False, method="min").astype(int)
    return frame.sort_values("holdout_pearson", ascending=False).reset_index(drop=True)


def search_blend_weights(
    y_valid: np.ndarray,
    pred_map: dict[str, np.ndarray],
    step: float = 0.1,
) -> tuple[dict[str, float], dict[str, float]]:
    names = list(pred_map.keys())
    best_metrics = {"pearson": -1.0}
    best_weights = {name: 0.0 for name in names}
    weight_grid = np.arange(0.0, 1.0 + step / 2, step)
    if len(names) == 2:
        combos = [(a, 1.0 - a) for a in weight_grid]
    else:
        combos = []
        for w1 in weight_grid:
            for w2 in weight_grid:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                if w3 < 0:
                    w3 = 0.0
                if abs(w1 + w2 + w3 - 1.0) <= 1e-6:
                    combos.append((w1, w2, w3))
    for combo in combos:
        blend = np.zeros_like(y_valid, dtype=np.float64)
        weights = {names[i]: float(combo[i]) for i in range(len(names))}
        for name, weight in weights.items():
            blend += weight * pred_map[name]
        metrics = evaluate_regression(y_valid, blend)
        if metrics["pearson"] > best_metrics["pearson"]:
            best_metrics = metrics
            best_weights = weights
    return best_weights, best_metrics


def run_step3_tree_blend(
    root: Path,
    output_dir: Path,
    submission_dir: Path,
    data: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, float | int],
    log_path: Path,
    *,
    sample_rows: int | None = None,
    smoke_test: bool = False,
) -> ExperimentRecord:
    log_message(log_path, "Step3: LGBM + CatBoost + XGBoost 集成开始")
    train_idx, valid_idx = time_order_split(len(data), valid_fraction=0.2, gap_rows=0)
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)

    args = make_lgbm_args(root, params, smoke_test=smoke_test)
    lgbm_model, lgbm_valid_pred, lgbm_metrics = train_lgbm_model(data, feature_cols, train_idx, valid_idx, args)
    assert lgbm_valid_pred is not None
    cat_model, cat_valid_pred, cat_metrics = train_catboost_holdout(
        data,
        feature_cols,
        train_idx,
        valid_idx,
        smoke_test=smoke_test,
    )
    xgb_valid_pred, xgb_metrics, xgb_model = train_xgb_holdout(
        data,
        feature_cols,
        train_idx,
        valid_idx,
        smoke_test=smoke_test,
    )

    pred_map = {
        "lgbm": lgbm_valid_pred,
        "catboost": cat_valid_pred,
        "xgboost": xgb_valid_pred,
    }
    weights, blend_metrics = search_blend_weights(y_valid, pred_map, step=0.1)
    blend_valid = (
        weights["lgbm"] * lgbm_valid_pred
        + weights["catboost"] * cat_valid_pred
        + weights["xgboost"] * xgb_valid_pred
    )
    weight_frame = pd.DataFrame([weights | blend_metrics])
    weight_frame.to_csv(output_dir / "step3_blend_weight_search.csv", index=False)

    test_data = load_parquet_frame(root, "test.parquet", sample_rows=sample_rows if smoke_test else None, include_label=False)
    test_data = add_basic_market_features(test_data)
    test_lgbm = lgbm_model.predict(test_data[feature_cols], num_iteration=lgbm_model.best_iteration)
    test_cat = cat_model.predict(test_data[feature_cols]).astype(np.float64)
    test_xgb = xgb_model.predict(test_data[feature_cols]).astype(np.float64)
    test_blend = (
        weights["lgbm"] * test_lgbm + weights["catboost"] * test_cat + weights["xgboost"] * test_xgb
    )
    submission_path = save_submission(root, test_blend, "submission_overnight_step3_tree_blend.csv", submission_dir)
    np.save(output_dir / "step3_valid_pred.npy", blend_valid)

    record = ExperimentRecord(
        step="step3",
        experiment="tree_triple_blend",
        holdout_pearson=blend_metrics["pearson"],
        holdout_rmse=blend_metrics["rmse"],
        holdout_mae=blend_metrics["mae"],
        vs_baseline=blend_metrics["pearson"] - BASELINE_PEARSON,
        beats_baseline=blend_metrics["pearson"] > BASELINE_PEARSON,
        submission_file=display_path(root, submission_path),
        notes=f"weights={weights}; single lgbm={lgbm_metrics['pearson']:.4f}, cat={cat_metrics['pearson']:.4f}, xgb={xgb_metrics['pearson']:.4f}",
        extra={"weights": weights},
    )
    log_message(log_path, f"Step3 完成 holdout pearson={blend_metrics['pearson']:.6f}")
    return record


def run_step4_temporal_features(
    root: Path,
    output_dir: Path,
    submission_dir: Path,
    params: dict[str, float | int],
    log_path: Path,
    *,
    sample_rows: int | None = None,
    smoke_test: bool = False,
) -> ExperimentRecord:
    log_message(log_path, "Step4: 时序扩展特征 LGBM 开始")
    data, feature_cols = prepare_dataset(root, temporal=True, sample_rows=sample_rows)
    save_json({"feature_columns": feature_cols}, output_dir / "step4_temporal_features.json")
    train_idx, valid_idx = time_order_split(len(data), valid_fraction=0.2, gap_rows=0)
    args = make_lgbm_args(root, params, smoke_test=smoke_test)
    model, valid_pred, metrics = train_lgbm_model(data, feature_cols, train_idx, valid_idx, args)
    assert valid_pred is not None
    save_lgbm_model(model, output_dir / "step4_temporal_lgbm.txt")

    test_data = load_parquet_frame(root, "test.parquet", sample_rows=sample_rows if smoke_test else None, include_label=False)
    test_data = add_temporal_features(test_data)
    test_pred = model.predict(test_data[feature_cols], num_iteration=model.best_iteration)
    submission_path = save_submission(root, test_pred, "submission_overnight_step4_temporal.csv", submission_dir)

    record = ExperimentRecord(
        step="step4",
        experiment="temporal_extended_lgbm",
        holdout_pearson=metrics["pearson"],
        holdout_rmse=metrics["rmse"],
        holdout_mae=metrics["mae"],
        vs_baseline=metrics["pearson"] - BASELINE_PEARSON,
        beats_baseline=metrics["pearson"] > BASELINE_PEARSON,
        submission_file=display_path(root, submission_path),
        notes=f"新增特征数={len(feature_cols) - 792}",
        extra={"feature_count": len(feature_cols)},
    )
    log_message(log_path, f"Step4 完成 holdout pearson={metrics['pearson']:.6f}")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="树模型集成与时序扩展 LightGBM 流水线")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--steps", default="3,4", help="逗号分隔步骤编号；仅支持 3,4")
    parser.add_argument("--sample-rows", type=int, default=None, help="只读取前 N 行做烟测")
    parser.add_argument("--smoke-test", action="store_true", help="缩短树模型迭代并将提交写入 output-dir")
    parser.add_argument("--output-dir", default=None, help="实验输出目录；默认 outputs/experiments/overnight")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir) if args.output_dir else root / "outputs" / "experiments" / "overnight"
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = ensure_dir(output_dir)
    submission_dir = ensure_dir(output_dir / "submissions") if args.output_dir else ensure_dir(root / "outputs" / "submissions")
    log_path = output_dir / "overnight_progress.log"
    steps = {int(item.strip()) for item in args.steps.split(",") if item.strip()}
    unsupported_steps = steps - {3, 4}
    if unsupported_steps:
        raise ValueError(f"仅支持保留方法 steps=3,4，不再支持: {sorted(unsupported_steps)}")
    if not steps:
        raise ValueError("至少需要指定一个步骤: 3 或 4")

    log_message(log_path, f"夜间优化开始 steps={sorted(steps)} baseline={BASELINE_PEARSON:.6f}")
    params = load_best_lgbm_params(root)
    records: list[ExperimentRecord] = []

    if 3 in steps:
        data, feature_cols = prepare_dataset(root, temporal=False, sample_rows=args.sample_rows)
        records.append(
            run_step3_tree_blend(
                root,
                output_dir,
                submission_dir,
                data,
                feature_cols,
                params,
                log_path,
                sample_rows=args.sample_rows,
                smoke_test=args.smoke_test,
            )
        )
    if 4 in steps:
        records.append(
            run_step4_temporal_features(
                root,
                output_dir,
                submission_dir,
                params,
                log_path,
                sample_rows=args.sample_rows,
                smoke_test=args.smoke_test,
            )
        )

    summary = finalize_records(records)
    summary_path = output_dir / "optimization_summary.csv"
    summary.to_csv(summary_path, index=False)
    best_new = summary[(summary["step"] != "baseline") & (summary["beats_baseline"])]
    best_candidate = summary[summary["step"] != "baseline"].iloc[0] if len(summary[summary["step"] != "baseline"]) else None
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_pearson": BASELINE_PEARSON,
        "baseline_submission": "outputs/submissions/submission.csv",
        "records": summary.to_dict(orient="records"),
        "best_new_experiment": None,
    }
    if best_candidate is not None and bool(best_candidate["beats_baseline"]):
        src = Path(str(best_candidate["submission_file"]))
        if not src.is_absolute():
            src = root / src
        dst = submission_dir / "submission_overnight_best.csv"
        dst.write_bytes(src.read_bytes())
        payload["best_new_experiment"] = best_candidate.to_dict()
        payload["best_new_submission"] = display_path(root, dst)
        log_message(log_path, f"新的最优候选已复制到 {dst} (pearson={best_candidate['holdout_pearson']:.6f})")
    else:
        log_message(log_path, "暂无超过 baseline 的新结果；主提交 submission.csv 保持不变")

    save_json(payload, output_dir / "optimization_summary.json")
    log_message(log_path, f"汇总已保存: {summary_path}")

    plot_script = root / "src" / "visualization" / "plot_optimization_summary.py"
    if plot_script.is_file() and args.output_dir is None:
        import subprocess

        subprocess.run(
            [sys.executable, str(plot_script), "--root", str(root)],
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
