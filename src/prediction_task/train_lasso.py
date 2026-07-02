from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
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
    load_parquet_frame,
    save_json,
    validate_no_missing_or_infinite,
)
from prediction_task.metrics import evaluate_regression
from prediction_task.splits import time_order_split
from prediction_task.train_baseline import update_model_compare


DEFAULT_ALPHAS = "1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2"


def parse_alpha_grid(value: str) -> list[float]:
    alphas = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        alpha = float(item)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha 必须是正的有限数值: {item}")
        alphas.append(alpha)

    if not alphas:
        raise ValueError("alpha 网格不能为空")
    return alphas


def is_better_result(candidate: dict[str, float], current: dict[str, float] | None) -> bool:
    if current is None:
        return np.isfinite(candidate["pearson"])
    if not np.isfinite(candidate["pearson"]):
        return False
    if candidate["pearson"] > current["pearson"]:
        return True
    return candidate["pearson"] == current["pearson"] and candidate["rmse"] < current["rmse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练官方预测任务 Lasso Regression 对比模型")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--sample-rows", type=int, default=None, help="只读取前 N 行做小样本验证")
    parser.add_argument("--valid-fraction", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--gap-rows", type=int, default=0, help="训练集和验证集之间的 gap 行数")
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS, help="逗号分隔的 Lasso alpha 网格")
    parser.add_argument("--max-iter", type=int, default=5000, help="每个 alpha 的最大迭代次数")
    parser.add_argument("--tol", type=float, default=1e-4, help="Lasso 收敛容忍度")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    model_dir = ensure_dir(root / "models")
    alphas = parse_alpha_grid(args.alphas)

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    data = add_basic_market_features(data)
    feature_cols = get_feature_columns(data.columns)
    validate_no_missing_or_infinite(data, feature_cols + [TARGET_COL], context="Lasso 训练数据")

    train_idx, valid_idx = time_order_split(
        len(data),
        valid_fraction=args.valid_fraction,
        gap_rows=args.gap_rows,
    )
    X_train = data.iloc[train_idx][feature_cols]
    y_train = data.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float64)
    X_valid = data.iloc[valid_idx][feature_cols]
    y_valid = data.iloc[valid_idx][TARGET_COL].to_numpy(dtype=np.float64)

    alpha_results: list[dict[str, object]] = []
    best_result: dict[str, object] | None = None
    best_model: Pipeline | None = None
    best_valid_pred: np.ndarray | None = None

    for alpha in alphas:
        print(f"开始训练 Lasso alpha={alpha}")
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("lasso", Lasso(alpha=alpha, max_iter=args.max_iter, tol=args.tol)),
            ]
        )
        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)
        metrics = evaluate_regression(y_valid, valid_pred)
        lasso = model.named_steps["lasso"]
        result: dict[str, object] = {
            "model": "lasso",
            "alpha": alpha,
            "sample_rows": args.sample_rows or len(data),
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "valid_fraction": args.valid_fraction,
            "gap_rows": args.gap_rows,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "n_iter": int(lasso.n_iter_),
            "nonzero_coef_count": int(np.count_nonzero(lasso.coef_)),
            **metrics,
        }
        alpha_results.append(result)
        print(
            json.dumps(
                {
                    "alpha": alpha,
                    "pearson": metrics["pearson"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "n_iter": int(lasso.n_iter_),
                    "nonzero_coef_count": int(np.count_nonzero(lasso.coef_)),
                },
                ensure_ascii=False,
            )
        )

        if is_better_result(result, best_result):
            best_result = result
            best_model = model
            best_valid_pred = valid_pred

    if best_result is None or best_model is None or best_valid_pred is None:
        pd.DataFrame(alpha_results).to_csv(output_dir / "official_lasso_alpha_results.csv", index=False)
        raise ValueError("所有 alpha 的验证集 Pearson 均不是有限数值，无法选择最佳 Lasso 模型")

    for result in alpha_results:
        result["is_best"] = result["alpha"] == best_result["alpha"]

    lasso_step = best_model.named_steps["lasso"]
    coef_data = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": lasso_step.coef_,
        }
    )
    coef_data["abs_coefficient"] = coef_data["coefficient"].abs()
    coef_data["is_nonzero"] = coef_data["coefficient"] != 0.0
    coef_data = coef_data.sort_values("abs_coefficient", ascending=False)

    model_path = model_dir / "official_lasso.pkl"
    feature_path = model_dir / "official_lasso_features.json"
    joblib.dump(best_model, model_path)
    save_json({"feature_columns": feature_cols}, feature_path)

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        **best_result,
        "selected_alpha": best_result["alpha"],
        "tried_alphas": ",".join(str(alpha) for alpha in alphas),
        "generated_at": generated_at,
    }
    pd.DataFrame([summary]).to_csv(output_dir / "official_lasso_results.csv", index=False)
    pd.DataFrame(alpha_results).to_csv(output_dir / "official_lasso_alpha_results.csv", index=False)
    pd.DataFrame(
        {
            "row_index": valid_idx,
            "y_true": y_valid,
            "y_pred": best_valid_pred,
            "model": "lasso",
        }
    ).to_csv(output_dir / "official_lasso_valid_predictions.csv", index=False)
    coef_data.to_csv(output_dir / "official_lasso_coefficients.csv", index=False)
    update_model_compare(output_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"模型已保存: {model_path}")
    print(f"Lasso 系数已保存: {output_dir / 'official_lasso_coefficients.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
