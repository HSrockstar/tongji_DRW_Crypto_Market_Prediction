from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.second_place_features import (  # noqa: E402
    DEFAULT_ASSET_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_RAW_DATA_DIR,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_FILTERED_ROWS,
    EXPECTED_TEST_ROWS,
    FEATURE_LIST_NAME,
    PROJECT_ROOT,
    TEST_CACHE_NAME,
    TIME_FOLDS,
    TRAIN_CACHE_NAME,
    FileLogger,
    add_second_place_features,
    ensure_features_exist,
    filter_training_rows_by_time,
    finite_frame_values,
    load_feature_spec,
    read_json,
    read_parquet_frame,
    reduce_memory_usage,
    resolve_path,
    sha256_file,
    write_json,
)
from prediction_task.metrics import mae, pearson_corr, rmse  # noqa: E402


warnings.filterwarnings("ignore")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "second_place"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "second_place"
DEFAULT_SUBMISSION_DIR = PROJECT_ROOT / "outputs" / "submissions"
MODEL_NAMES = ["linear", "ridge", "lightgbm"]
CV_MODEL_NAMES = ["ridge", "lightgbm"]

RIDGE_PARAMS = {"alpha": 100.0}
LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 3,
    "min_data_in_leaf": 1000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 10.0,
    "n_estimators": 5000,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def parse_model_names(model_text: str) -> list[str]:
    models = [item.strip().lower() for item in model_text.split(",") if item.strip()]
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"未知模型名称: {unknown}")
    return models


def make_model(model_name: str, n_estimators: int | None = None) -> Any:
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(**RIDGE_PARAMS)),
            ]
        )
    if model_name == "lightgbm":
        params = LIGHTGBM_PARAMS.copy()
        if n_estimators is not None:
            params["n_estimators"] = int(n_estimators)
        return lgb.LGBMRegressor(**params)
    raise ValueError(f"未知模型名称: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行第 2 名迁移版线性/Ridge/LightGBM 方法")
    parser.add_argument("--raw-data-dir", default=str(DEFAULT_RAW_DATA_DIR), help="原始数据目录")
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR), help="迁移版资产目录")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="450 特征缓存目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="实验输出目录")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="模型输出目录")
    parser.add_argument("--submission-dir", default=str(DEFAULT_SUBMISSION_DIR), help="提交文件输出目录")
    parser.add_argument("--models", default="linear,ridge,lightgbm", help="逗号分隔模型列表")
    parser.add_argument("--max-folds", type=int, default=None, help="完整 CV 仅运行前 N 个 fold")
    parser.add_argument("--make-submissions", action="store_true", help="生成提交文件")
    parser.add_argument("--no-cv", action="store_true", help="跳过 Ridge/LightGBM CV，仅生成提交")
    parser.add_argument("--save-oof", dest="save_oof", action="store_true", default=True)
    parser.add_argument("--no-save-oof", dest="save_oof", action="store_false")
    parser.add_argument("--smoke-test", action="store_true", help="轻量路径检查，不执行固定时间窗口复现")
    parser.add_argument("--linear-smoke-rows", type=int, default=20_000, help="线性 smoke 使用的 raw 前 N 行")
    return parser.parse_args()


def load_cache(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_info = read_json(cache_dir / FEATURE_LIST_NAME)
    features = feature_info["features"]
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"缓存特征数量不符合预期: {len(features)}")
    train_df = pd.read_parquet(cache_dir / TRAIN_CACHE_NAME)
    test_df = pd.read_parquet(cache_dir / TEST_CACHE_NAME)
    if len(train_df) != EXPECTED_FILTERED_ROWS:
        raise ValueError(f"训练缓存行数不符合预期: {len(train_df)}")
    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(f"测试缓存行数不符合预期: {len(test_df)}")
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    return train_df, test_df, features


def load_cache_for_smoke(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_info = read_json(cache_dir / FEATURE_LIST_NAME)
    features = feature_info["features"]
    train_df = pd.read_parquet(cache_dir / TRAIN_CACHE_NAME)
    test_df = pd.read_parquet(cache_dir / TEST_CACHE_NAME)
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    return train_df, test_df, features


def validate_fold_masks(train_df: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    train_start = pd.Timestamp(fold["train_start"])
    valid_start = pd.Timestamp(fold["valid_start"])
    valid_end = pd.Timestamp(fold["valid_end"])
    train_mask = (train_df["timestamp"] >= train_start) & (train_df["timestamp"] < valid_start)
    valid_mask = (train_df["timestamp"] >= valid_start) & (train_df["timestamp"] < valid_end)
    if not train_mask.any() or not valid_mask.any():
        raise ValueError(f"fold {fold['fold']} 训练或验证样本为空")
    if train_df.loc[train_mask, "timestamp"].max() >= train_df.loc[valid_mask, "timestamp"].min():
        raise ValueError(f"fold {fold['fold']} 存在时间泄露")
    return train_mask, valid_mask


def fit_predict_fold(
    model_name: str,
    fold: dict[str, Any],
    train_df: pd.DataFrame,
    features: list[str],
    logger: FileLogger,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_mask, valid_mask = validate_fold_masks(train_df, fold)
    x_train = train_df.loc[train_mask, features].to_numpy(dtype=np.float32)
    y_train = train_df.loc[train_mask, "label"].to_numpy(dtype=np.float32)
    x_valid = train_df.loc[valid_mask, features].to_numpy(dtype=np.float32)
    y_valid = train_df.loc[valid_mask, "label"].to_numpy(dtype=np.float32)
    if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
        raise ValueError(f"{model_name} fold {fold['fold']} 特征包含 NaN/Inf")

    logger.write(f"训练 {model_name} fold {fold['fold']}: train={len(x_train)}, valid={len(x_valid)}")
    model = make_model(model_name)
    fit_start = time.perf_counter()
    if model_name == "lightgbm":
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_valid), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{model_name} fold {fold['fold']} 预测包含 NaN/Inf")

    fold_result = {
        "model": model_name,
        "fold": int(fold["fold"]),
        "fold_label": fold["label"],
        "train_start": fold["train_start"],
        "valid_start": fold["valid_start"],
        "valid_end": fold["valid_end"],
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "pearson": pearson_corr(y_valid, prediction),
        "rmse": rmse(y_valid, prediction),
        "mae": mae(y_valid, prediction),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "best_iteration": getattr(model, "best_iteration_", None),
        "elapsed_seconds": round(time.perf_counter() - fit_start, 3),
    }
    oof = pd.DataFrame(
        {
            "timestamp": train_df.loc[valid_mask, "timestamp"].to_numpy(),
            "label": y_valid,
            "prediction": prediction,
            "fold": int(fold["fold"]),
            "fold_label": fold["label"],
        }
    )
    logger.write(f"{model_name} fold {fold['fold']} 完成: Pearson={fold_result['pearson']:.6f}")
    return fold_result, oof


def run_smoke_cv(model_name: str, train_df: pd.DataFrame, features: list[str], logger: FileLogger) -> dict[str, Any]:
    valid_size = max(1, int(round(len(train_df) * 0.2)))
    train_end = len(train_df) - valid_size
    x_train = train_df.iloc[:train_end][features].to_numpy(dtype=np.float32)
    y_train = train_df.iloc[:train_end]["label"].to_numpy(dtype=np.float32)
    x_valid = train_df.iloc[train_end:][features].to_numpy(dtype=np.float32)
    y_valid = train_df.iloc[train_end:]["label"].to_numpy(dtype=np.float32)
    n_estimators = 20 if model_name == "lightgbm" else None
    model = make_model(model_name, n_estimators=n_estimators)
    logger.write(f"smoke 训练 {model_name}: train={len(x_train)}, valid={len(x_valid)}")
    if model_name == "lightgbm":
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], callbacks=[lgb.log_evaluation(0)])
    else:
        model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_valid), dtype=np.float64)
    return {
        "model": model_name,
        "fold": 1,
        "fold_label": "smoke_last_20_percent",
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "pearson": pearson_corr(y_valid, prediction),
        "rmse": rmse(y_valid, prediction),
        "mae": mae(y_valid, prediction),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "best_iteration": getattr(model, "best_iteration_", None),
        "elapsed_seconds": np.nan,
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, group in results.groupby("model", sort=False):
        pearson_std = float(group["pearson"].std(ddof=0))
        pearson_mean = float(group["pearson"].mean())
        best_iter = group["best_iteration"].dropna()
        rows.append(
            {
                "model": model_name,
                "fold_count": int(len(group)),
                "pearson_mean": pearson_mean,
                "pearson_std": pearson_std,
                "pearson_min": float(group["pearson"].min()),
                "pearson_max": float(group["pearson"].max()),
                "rmse_mean": float(group["rmse"].mean()),
                "mae_mean": float(group["mae"].mean()),
                "stability_score": pearson_mean - 0.5 * pearson_std,
                "best_iteration_median": float(best_iter.median()) if not best_iter.empty else np.nan,
                "best_iteration_mean": float(best_iter.mean()) if not best_iter.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("stability_score", ascending=False).reset_index(drop=True)


def run_cv(
    train_df: pd.DataFrame,
    features: list[str],
    models: list[str],
    output_dir: Path,
    logger: FileLogger,
    max_folds: int | None,
    save_oof: bool,
    smoke_test: bool,
) -> pd.DataFrame:
    cv_models = [model for model in models if model in CV_MODEL_NAMES]
    results: list[dict[str, Any]] = []
    if not cv_models:
        return pd.DataFrame()
    (output_dir / "oof").mkdir(parents=True, exist_ok=True)
    folds = TIME_FOLDS[:max_folds] if max_folds else TIME_FOLDS
    for model_name in cv_models:
        if smoke_test:
            results.append(run_smoke_cv(model_name, train_df, features, logger))
            continue
        oof_frames: list[pd.DataFrame] = []
        for fold in folds:
            fold_result, oof = fit_predict_fold(model_name, fold, train_df, features, logger)
            results.append(fold_result)
            oof_frames.append(oof)
        if save_oof:
            oof_all = pd.concat(oof_frames, axis=0, ignore_index=True)
            oof_path = output_dir / "oof" / f"oof_predictions_{model_name}.csv"
            oof_all.to_csv(oof_path, index=False)
            logger.write(f"{model_name} OOF 预测已保存: {oof_path}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "cv_results.csv", index=False)
    summary_df = summarize_results(results_df)
    summary_df.to_csv(output_dir / "cv_summary.csv", index=False)
    write_json(model_params_for_output(), output_dir / "model_params.json")
    return summary_df


def lightgbm_final_estimators(cv_results_path: Path) -> int | None:
    if not cv_results_path.is_file():
        return None
    cv_results = pd.read_csv(cv_results_path)
    if "best_iteration" not in cv_results.columns:
        return None
    lightgbm_iters = cv_results.loc[cv_results["model"] == "lightgbm", "best_iteration"].dropna()
    lightgbm_iters = lightgbm_iters[lightgbm_iters > 0]
    if lightgbm_iters.empty:
        return None
    return max(1, int(round(float(lightgbm_iters.median()))))


def write_submission(
    model_name: str,
    prediction: np.ndarray,
    sample_submission: pd.DataFrame,
    submission_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    submission = sample_submission.copy()
    if len(submission) != len(prediction):
        raise ValueError(f"提交模板行数与预测行数不一致: {len(submission)} != {len(prediction)}")
    submission["prediction"] = prediction
    submission_path = submission_dir / f"submission_second_place_{model_name}.csv"
    submission.to_csv(submission_path, index=False)
    return submission_path, {
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "prediction_has_nan_or_inf": bool(not np.isfinite(prediction).all()),
        "submission_sha256": sha256_file(submission_path),
    }


def generate_cache_model_submission(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    raw_data_dir: Path,
    output_dir: Path,
    model_dir: Path,
    submission_dir: Path,
    logger: FileLogger,
    smoke_test: bool,
) -> dict[str, Any]:
    x_train = train_df[features].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    x_test = test_df[features].to_numpy(dtype=np.float32)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError(f"{model_name} 训练或测试特征包含 NaN/Inf")

    n_estimators = None
    if model_name == "lightgbm":
        n_estimators = 20 if smoke_test else lightgbm_final_estimators(output_dir / "cv_results.csv")
        if n_estimators is not None:
            logger.write(f"LightGBM 最终训练使用 n_estimators={n_estimators}")
    model = make_model(model_name, n_estimators=n_estimators)
    logger.write(f"在全部筛选训练样本上重训模型: {model_name}")
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_test), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{model_name} 预测结果包含 NaN/Inf")

    sample_submission = pd.read_csv(raw_data_dir / "sample_submission.csv")
    if smoke_test:
        sample_submission = sample_submission.head(len(prediction)).copy()
    submission_path, prediction_summary = write_submission(model_name, prediction, sample_submission, submission_dir)
    model_path = model_dir / f"second_place_{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.write(f"{model_name} submission 已保存: {submission_path}")
    return {
        "model": model_name,
        "model_path": str(model_path),
        "submission_path": str(submission_path),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(features),
        "lightgbm_final_n_estimators": n_estimators,
        **prediction_summary,
    }


def generate_linear_submission(
    raw_data_dir: Path,
    asset_dir: Path,
    output_dir: Path,
    model_dir: Path,
    submission_dir: Path,
    logger: FileLogger,
    smoke_rows: int | None,
) -> dict[str, Any]:
    spec = load_feature_spec(asset_dir)
    features = spec["final_features"]
    logger.write("线性方案读取 train.parquet")
    train_df = read_parquet_frame(raw_data_dir / "train.parquet", smoke_rows)
    logger.write("线性方案读取 test.parquet")
    test_df = read_parquet_frame(raw_data_dir / "test.parquet", smoke_rows)
    sample_submission = pd.read_csv(raw_data_dir / "sample_submission.csv")
    if smoke_rows is not None:
        sample_submission = sample_submission.head(len(test_df)).copy()

    train_df = reduce_memory_usage(train_df, logger.write, "linear_train")
    test_df = reduce_memory_usage(test_df, logger.write, "linear_test")
    logger.write("线性方案执行公开市场特征工程")
    train_df = add_second_place_features(train_df, spec)
    test_df = add_second_place_features(test_df, spec)
    logger.write("线性方案执行时间过滤")
    train_clean = filter_training_rows_by_time(train_df, asset_dir)
    if train_clean.empty:
        raise ValueError("线性方案时间过滤后训练集为空；请增大 --linear-smoke-rows")

    ensure_features_exist(train_clean, features, "线性训练集")
    ensure_features_exist(test_df, features, "线性测试集")
    finite_frame_values(train_clean, features, "线性训练集")
    finite_frame_values(test_df, features, "线性测试集")

    model = make_model("linear")
    logger.write("训练 sklearn.linear_model.LinearRegression 默认模型")
    x_train = train_clean[features].values
    y_train = train_clean["label"]
    x_test = test_df[features].values
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_test), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("线性方案预测结果包含 NaN/Inf")

    submission_path, prediction_summary = write_submission("linear", prediction, sample_submission, submission_dir)
    model_path = model_dir / "second_place_linear.joblib"
    joblib.dump(model, model_path)
    logger.write(f"linear submission 已保存: {submission_path}")
    feature_info = {
        "negative_features_count": len(spec["negative_features"]),
        "positive_features_count": len(spec["positive_features"]),
        "final_features_count": len(features),
        "final_features": features,
    }
    write_json(feature_info, output_dir / "linear_feature_list.json")
    return {
        "model": "linear",
        "model_path": str(model_path),
        "submission_path": str(submission_path),
        "train_rows_loaded": int(len(train_df)),
        "test_rows_loaded": int(len(test_df)),
        "train_rows_after_time_filter": int(len(train_clean)),
        "feature_count": len(features),
        **prediction_summary,
    }


def model_params_for_output() -> dict[str, Any]:
    return {
        "models": {
            "linear": "sklearn.linear_model.LinearRegression()",
            "ridge": RIDGE_PARAMS,
            "lightgbm": LIGHTGBM_PARAMS,
        },
        "time_folds": TIME_FOLDS,
        "stability_score": "pearson_mean - 0.5 * pearson_std",
    }


def main() -> int:
    args = parse_args()
    raw_data_dir = resolve_path(args.raw_data_dir, DEFAULT_RAW_DATA_DIR)
    asset_dir = resolve_path(args.asset_dir, DEFAULT_ASSET_DIR)
    cache_dir = resolve_path(args.cache_dir, DEFAULT_CACHE_DIR)
    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    model_dir = resolve_path(args.model_dir, DEFAULT_MODEL_DIR)
    submission_dir = resolve_path(args.submission_dir, DEFAULT_SUBMISSION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    logger = FileLogger(output_dir / ("run_log_smoke.txt" if args.smoke_test else "run_log.txt"))
    start = time.perf_counter()

    models = parse_model_names(args.models)
    if args.max_folds is not None and args.max_folds <= 0:
        raise ValueError("--max-folds 必须为正整数")
    logger.write(f"模型列表: {models}")
    logger.write(f"缓存目录: {cache_dir}")
    logger.write(f"输出目录: {output_dir}")

    train_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None
    features: list[str] | None = None
    if any(model in CV_MODEL_NAMES for model in models):
        if args.smoke_test:
            train_df, test_df, features = load_cache_for_smoke(cache_dir)
        else:
            train_df, test_df, features = load_cache(cache_dir)
        if not args.no_cv:
            run_cv(
                train_df,
                features,
                models,
                output_dir,
                logger,
                args.max_folds,
                args.save_oof,
                args.smoke_test,
            )

    submission_summaries: dict[str, Any] = {}
    if args.make_submissions:
        if "linear" in models:
            linear_smoke_rows = args.linear_smoke_rows if args.smoke_test else None
            submission_summaries["linear"] = generate_linear_submission(
                raw_data_dir,
                asset_dir,
                output_dir,
                model_dir,
                submission_dir,
                logger,
                linear_smoke_rows,
            )
        for model_name in [model for model in models if model in CV_MODEL_NAMES]:
            if train_df is None or test_df is None or features is None:
                if args.smoke_test:
                    train_df, test_df, features = load_cache_for_smoke(cache_dir)
                else:
                    train_df, test_df, features = load_cache(cache_dir)
            submission_summaries[model_name] = generate_cache_model_submission(
                model_name,
                train_df,
                test_df,
                features,
                raw_data_dir,
                output_dir,
                model_dir,
                submission_dir,
                logger,
                args.smoke_test,
            )

    run_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "models": models,
        "smoke_test": bool(args.smoke_test),
        "make_submissions": bool(args.make_submissions),
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
        "submission_dir": str(submission_dir),
        "submission_summaries": submission_summaries,
    }
    write_json(run_summary, output_dir / ("run_summary_smoke.json" if args.smoke_test else "run_summary.json"))
    logger.write(f"第 2 名迁移版运行完成，总耗时 {run_summary['elapsed_seconds']:.1f} 秒")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
