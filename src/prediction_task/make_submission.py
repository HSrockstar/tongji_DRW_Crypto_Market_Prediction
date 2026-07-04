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

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import add_basic_market_features
from data_preprocessing.preprocess import (  # noqa: E402
    DEFAULT_ROOT,
    ensure_dir,
    load_json,
    load_parquet_frame,
    raw_path,
    save_json,
    validate_no_missing_or_infinite,
)


def load_lgbm_model(model_path: Path) -> lgb.Booster:
    if model_path.suffix.lower() == ".txt":
        model_string = model_path.read_text(encoding="utf-8")
        try:
            return lgb.Booster(model_str=model_string)
        except lgb.basic.LightGBMError:
            # 部分 LightGBM 版本在 tree_sizes 快速加载路径上会误解析，去掉后可走顺序解析。
            cleaned_model_string = "\n".join(
                line for line in model_string.splitlines() if not line.startswith("tree_sizes=")
            )
            return lgb.Booster(model_str=cleaned_model_string)
    try:
        return lgb.Booster(model_file=str(model_path))
    except lgb.basic.LightGBMError:
        model_string = model_path.read_text(encoding="utf-8")
        return lgb.Booster(model_str=model_string)


def infer_feature_path(model_path: Path) -> Path:
    if model_path.name == "official_lgbm.txt":
        return model_path.with_name("official_lgbm_features.json")
    if model_path.name == "official_lgbm_selected.txt":
        return model_path.with_name("official_lgbm_selected_features.json")
    if model_path.name == "selected_lgbm.txt":
        return model_path.with_name("selected_lgbm_features.json")
    if model_path.name == "official_ridge.pkl":
        return model_path.with_name("official_ridge_features.json")
    if model_path.name == "official_ridge_selected.pkl":
        return model_path.with_name("official_ridge_selected_features.json")
    if model_path.name == "selected_ridge.pkl":
        return model_path.with_name("selected_ridge_features.json")
    return model_path.with_suffix("").with_name(f"{model_path.stem}_features.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成官方预测任务提交文件")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--model", required=True, help="模型文件路径")
    parser.add_argument("--feature-file", default=None, help="特征列 JSON，默认按模型名推断")
    parser.add_argument("--model-type", choices=["auto", "ridge", "lightgbm"], default="auto")
    parser.add_argument("--smoke-rows", type=int, default=None, help="只预测前 N 行，生成 smoke submission")
    parser.add_argument("--output", default=None, help="提交文件输出路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    feature_path = Path(args.feature_file) if args.feature_file else infer_feature_path(model_path)
    if not feature_path.is_absolute():
        feature_path = root / feature_path

    feature_payload = load_json(feature_path)
    feature_cols = feature_payload["feature_columns"]
    test_data = load_parquet_frame(
        root,
        "test.parquet",
        sample_rows=args.smoke_rows,
        include_label=False,
    )
    test_data = add_basic_market_features(test_data)
    validate_no_missing_or_infinite(test_data, feature_cols, context="提交预测数据")
    X_test = test_data[feature_cols]

    model_type = args.model_type
    if model_type == "auto":
        if model_path.suffix.lower() == ".txt":
            model_type = "lightgbm"
        else:
            model_type = "ridge"

    if model_type == "lightgbm":
        model = load_lgbm_model(model_path)
        prediction = model.predict(X_test, num_iteration=model.best_iteration)
    else:
        model = joblib.load(model_path)
        prediction = model.predict(X_test)

    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("预测结果包含 NaN 或无穷值")

    sample_submission = pd.read_csv(raw_path(root, "sample_submission.csv"))
    if args.smoke_rows is not None:
        submission = sample_submission.head(args.smoke_rows).copy()
        default_output = root / "outputs" / "submissions" / "smoke_submission.csv"
    else:
        submission = sample_submission.copy()
        default_output = root / "outputs" / "submissions" / "submission.csv"

    if len(submission) != len(prediction):
        raise ValueError(f"提交行数与预测行数不一致: {len(submission)} != {len(prediction)}")
    submission["prediction"] = prediction

    output_path = Path(args.output) if args.output else default_output
    if not output_path.is_absolute():
        output_path = root / output_path
    ensure_dir(output_path.parent)
    submission.to_csv(output_path, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path),
        "model_type": model_type,
        "feature_file": str(feature_path),
        "output": str(output_path),
        "rows": int(len(submission)),
        "smoke_rows": args.smoke_rows,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "has_nan_or_inf": bool(not np.isfinite(prediction).all()),
    }
    save_json(summary, output_path.with_name(f"{output_path.stem}_summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
