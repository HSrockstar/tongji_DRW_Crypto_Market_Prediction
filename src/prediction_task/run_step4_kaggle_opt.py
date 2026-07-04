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
from data_preprocessing.temporal_features import add_step4_temporal_features
from prediction_task.metrics import evaluate_regression
from prediction_task.run_time_cv_experiments import fit_predict_lgbm
from prediction_task.splits import purged_group_time_series_splits, time_order_split
from prediction_task.train_lgbm import lgb_pearson_eval, save_lgbm_model

# 525886 行 ≈ 12 个月 (2023-03 ~ 2024-02)
ROWS_PER_MONTH = 525886 // 12
RECENT_WINDOWS = {
    "3mo": 3 * ROWS_PER_MONTH,
    "6mo": 6 * ROWS_PER_MONTH,
    "9mo": 9 * ROWS_PER_MONTH,
    "full": None,
}

KAGGLE_PARAM_PRESETS: list[dict[str, float | int | str]] = [
    {"name": "baseline", "num_leaves": 31, "min_data_in_leaf": 500, "lambda_l1": 0.0, "lambda_l2": 50.0, "feature_fraction": 0.7},
    {"name": "strong_reg", "num_leaves": 15, "min_data_in_leaf": 800, "lambda_l1": 0.0, "lambda_l2": 100.0, "feature_fraction": 0.6},
    {"name": "medium_reg", "num_leaves": 31, "min_data_in_leaf": 600, "lambda_l1": 1.0, "lambda_l2": 70.0, "feature_fraction": 0.65},
]

STEP4_KAGGLE_LB = {"public": 0.0408, "private": 0.0291}


def _plot_comparison(root: Path, summary_rows: list[dict], cv_frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = ensure_dir(root / "outputs" / "figures" / "prediction_task")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    if len(cv_frame) > 0:
        cv_frame = cv_frame.copy()
        cv_frame["label"] = cv_frame["window"] + "/" + cv_frame["preset"]
        plt.figure(figsize=(10, 5))
        sns.barplot(data=cv_frame.head(12), x="pearson_mean", y="label", color="#55A868")
        plt.xlabel("Purged CV Pearson mean")
        plt.title("Step4 Kaggle 对齐：近期窗口 × 参数 Purged CV")
        plt.tight_layout()
        plt.savefig(fig_dir / "step4_kaggle_cv_search.png", dpi=150)
        plt.close()

    labels = ["Purged CV", "Pre-window ref", "Forward valid", "Kaggle Public (step4)"]
    opt = summary_rows[1]
    orig = summary_rows[0]
    values = [
        opt.get("purged_cv_mean"),
        opt.get("pre_window_pearson"),
        opt.get("forward_valid_pearson"),
        orig.get("kaggle_public"),
    ]
    plt.figure(figsize=(8, 4))
    plot_df = pd.DataFrame({"metric": labels, "value": values})
    sns.barplot(data=plot_df, x="metric", y="value", hue="metric", dodge=False, palette="Set2")
    plt.title("Step4 原版 Kaggle vs 新方案验证指标")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "step4_kaggle_metric_compare.png", dpi=150)
    plt.close()

def log_message(log_path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_base_params(root: Path) -> dict[str, float | int]:
    return load_json(root / "outputs" / "experiments" / "lgbm_best_params.json")["params"]


def make_lgbm_args(params: dict[str, float | int], seed: int = 42) -> argparse.Namespace:
    return argparse.Namespace(
        root=".",
        sample_rows=None,
        valid_fraction=0.2,
        gap_rows=0,
        feature_file=None,
        learning_rate=0.01,
        num_boost_round=3000,
        early_stopping_rounds=200,
        metric_for_early_stop="pearson",
        bagging_fraction=0.9,
        max_depth=-1,
        min_gain_to_split=0.0,
        num_threads=0,
        log_period=200,
        seed=seed,
        num_leaves=int(params["num_leaves"]),
        min_data_in_leaf=int(params["min_data_in_leaf"]),
        lambda_l1=float(params["lambda_l1"]),
        lambda_l2=float(params["lambda_l2"]),
        feature_fraction=float(params["feature_fraction"]),
    )


def slice_recent(data: pd.DataFrame, window_rows: int | None) -> pd.DataFrame:
    if window_rows is None or window_rows >= len(data):
        return data.reset_index(drop=True)
    return data.iloc[-window_rows:].reset_index(drop=True)


def run_purged_cv(
    data: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, float | int],
) -> dict[str, float | list[float]]:
    splits = purged_group_time_series_splits(len(data), n_groups=6, gap_groups=1)
    args = make_lgbm_args(params)
    pearsons: list[float] = []
    for split in splits:
        valid_idx = split["valid_idx"]
        y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)
        valid_pred, _ = fit_predict_lgbm(data, split, feature_cols, args)
        pearsons.append(evaluate_regression(y_valid, valid_pred)["pearson"])
    arr = np.asarray(pearsons, dtype=np.float64)
    return {
        "pearson_mean": float(arr.mean()),
        "pearson_std": float(arr.std()),
        "fold_pearsons": pearsons,
    }


def train_final_model(
    data: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, float | int],
    *,
    valid_fraction: float = 0.15,
) -> tuple[lgb.Booster, np.ndarray, np.ndarray, dict[str, float]]:
    train_idx, valid_idx = time_order_split(len(data), valid_fraction=valid_fraction, gap_rows=0)
    args = make_lgbm_args(params)
    X_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    X_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)

    lgb_params = {
        "objective": "regression",
        "metric": "None",
        "learning_rate": 0.01,
        "num_leaves": int(params["num_leaves"]),
        "max_depth": -1,
        "min_data_in_leaf": int(params["min_data_in_leaf"]),
        "feature_fraction": float(params["feature_fraction"]),
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": float(params["lambda_l1"]),
        "lambda_l2": float(params["lambda_l2"]),
        "seed": 42,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_cols, reference=train_set)
    model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        feval=lgb_pearson_eval,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
    )
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    metrics = evaluate_regression(y_valid, valid_pred)
    return model, train_idx, valid_idx, metrics


def save_submission(root: Path, predictions: np.ndarray, file_name: str) -> Path:
    template = pd.read_csv(raw_path(root, "sample_submission.csv"))
    output = template.copy()
    output["prediction"] = np.asarray(predictions, dtype=np.float64)
    path = ensure_dir(root / "outputs" / "submissions") / file_name
    output.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step4 Kaggle 对齐优化：近期窗口 + Purged CV 选参")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--extended-features", action="store_true", default=True)
    parser.add_argument("--quick", action="store_true", help="只跑 3mo/6mo + baseline 参数")
    parser.add_argument("--skip-cv", action="store_true", help="跳过 CV，读取已有 window_param_purged_cv.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments" / "step4_kaggle_opt")
    log_path = output_dir / "step4_kaggle_opt.log"

    log_message(log_path, "Step4 Kaggle 对齐优化开始")
    log_message(log_path, f"参考 Kaggle LB (step4): public={STEP4_KAGGLE_LB['public']}, private={STEP4_KAGGLE_LB['private']}")

    data_full = load_parquet_frame(root, "train.parquet", include_label=True)
    data_full = add_step4_temporal_features(data_full, extended=True)
    feature_cols = get_feature_columns(data_full.columns)
    validate_no_missing_or_infinite(data_full, feature_cols + [TARGET_COL], context="step4 kaggle opt")
    save_json({"feature_columns": feature_cols, "feature_count": len(feature_cols)}, output_dir / "feature_columns.json")
    log_message(log_path, f"增强时序特征数: {len(feature_cols)} (base 792 + {len(feature_cols) - 792})")

    windows = RECENT_WINDOWS if not args.quick else {"3mo": RECENT_WINDOWS["3mo"], "6mo": RECENT_WINDOWS["6mo"]}
    presets = KAGGLE_PARAM_PRESETS if not args.quick else [KAGGLE_PARAM_PRESETS[0], KAGGLE_PARAM_PRESETS[1]]

    cv_rows: list[dict] = []
    if args.skip_cv and (output_dir / "window_param_purged_cv.csv").is_file():
        cv_frame = pd.read_csv(output_dir / "window_param_purged_cv.csv")
        log_message(log_path, "跳过 CV，加载已有 window_param_purged_cv.csv")
    else:
        for window_name, window_rows in windows.items():
            data = slice_recent(data_full, window_rows)
            log_message(log_path, f"窗口 {window_name}: {len(data)} 行")
            for preset in presets:
                params = {k: v for k, v in preset.items() if k != "name"}
                cv = run_purged_cv(data, feature_cols, params)
                row = {
                    "window": window_name,
                    "window_rows": len(data),
                    "preset": preset["name"],
                    **params,
                    **cv,
                }
                cv_rows.append(row)
                log_message(
                    log_path,
                    f"  CV {window_name}/{preset['name']} purged_mean={cv['pearson_mean']:.6f} std={cv['pearson_std']:.6f}",
                )

        cv_frame = pd.DataFrame(cv_rows).sort_values("pearson_mean", ascending=False).reset_index(drop=True)
        cv_frame.to_csv(output_dir / "window_param_purged_cv.csv", index=False)
    best = cv_frame.iloc[0]
    best_window = str(best["window"])
    best_preset = str(best["preset"])
    best_params = {
        "num_leaves": int(best["num_leaves"]),
        "min_data_in_leaf": int(best["min_data_in_leaf"]),
        "lambda_l1": float(best["lambda_l1"]),
        "lambda_l2": float(best["lambda_l2"]),
        "feature_fraction": float(best["feature_fraction"]),
    }
    log_message(
        log_path,
        f"Purged CV 最优: window={best_window}, preset={best_preset}, mean={best['pearson_mean']:.6f}",
    )

    best_window_rows = RECENT_WINDOWS.get(best_window)
    data_train = slice_recent(data_full, best_window_rows)
    model, train_idx, valid_idx, forward_metrics = train_final_model(
        data_train, feature_cols, best_params, valid_fraction=0.15
    )
    # 训练窗口之前的 20% 切片作参考验证（避免与 3mo 训练窗重叠造成泄露）
    pre_window_metrics: dict[str, float] | None = None
    if best_window_rows is not None and best_window_rows < len(data_full):
        pre_start = max(0, len(data_full) - best_window_rows - best_window_rows // 5)
        pre_end = len(data_full) - best_window_rows
        if pre_end > pre_start + 1000:
            pre_pred = model.predict(
                data_full.iloc[pre_start:pre_end][feature_cols],
                num_iteration=model.best_iteration,
            )
            pre_window_metrics = evaluate_regression(
                data_full.iloc[pre_start:pre_end][TARGET_COL].to_numpy(dtype=np.float64),
                pre_pred,
            )
            log_message(
                log_path,
                f"训练窗之前参考段 pearson={pre_window_metrics['pearson']:.6f} (不重叠, 可比)",
            )

    log_message(
        log_path,
        f"近期窗口 forward-valid pearson={forward_metrics['pearson']:.6f} (窗内末15%, 仅监控)",
    )

    save_lgbm_model(model, output_dir / "step4_kaggle_opt_lgbm.txt")
    save_json(
        {
            "best_window": best_window,
            "best_preset": best_preset,
            "best_params": best_params,
            "purged_cv_mean": float(best["pearson_mean"]),
            "purged_cv_std": float(best["pearson_std"]),
            "forward_valid": forward_metrics,
            "pre_window_reference": pre_window_metrics,
            "feature_count": len(feature_cols),
        },
        output_dir / "best_config.json",
    )

    test_data = load_parquet_frame(root, "test.parquet", include_label=False)
    test_data = add_step4_temporal_features(test_data, extended=True)
    test_pred = model.predict(test_data[feature_cols], num_iteration=model.best_iteration)
    submission_path = save_submission(root, test_pred, "submission_step4_kaggle_opt.csv")
    best_submission_path = save_submission(root, test_pred, "submission_step4_kaggle_best.csv")

    summary_rows = [
        {
            "experiment": "step4_original_kaggle_lb",
            "selection_metric": "kaggle_lb",
            "purged_cv_mean": None,
            "forward_valid_pearson": None,
            "full_holdout_pearson": 0.09628,
            "kaggle_public": STEP4_KAGGLE_LB["public"],
            "kaggle_private": STEP4_KAGGLE_LB["private"],
            "submission": "outputs/submissions/submission_overnight_step4_temporal.csv",
            "notes": "夜间 step4 基线",
        },
        {
            "experiment": "step4_kaggle_opt",
            "selection_metric": "purged_cv_on_recent_window",
            "window": best_window,
            "preset": best_preset,
            "purged_cv_mean": float(best["pearson_mean"]),
            "purged_cv_std": float(best["pearson_std"]),
            "forward_valid_pearson": forward_metrics["pearson"],
            "pre_window_pearson": pre_window_metrics["pearson"] if pre_window_metrics else None,
            "full_holdout_pearson": None,
            "kaggle_public": None,
            "kaggle_private": None,
            "submission": str(submission_path.relative_to(root)),
            "notes": f"extended temporal, {len(feature_cols)} features",
        },
    ]
    summary_path = output_dir / "step4_kaggle_comparison.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    _plot_comparison(root, summary_rows, cv_frame, output_dir)

    log_message(log_path, f"提交文件: {submission_path}")
    log_message(log_path, f"推荐上传: {best_submission_path}")
    log_message(log_path, "Step4 Kaggle 对齐优化完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
