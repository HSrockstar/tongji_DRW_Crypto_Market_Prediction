from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.preprocess import DEFAULT_ROOT, ensure_dir  # noqa: E402
from prediction_task.metrics import evaluate_regression  # noqa: E402


GROUP_OVERALL = "整体"
GROUP_EXTREME = "异常波动"
MODEL_LABELS = {
    "ridge": "Ridge",
    "lightgbm": "LightGBM",
    "weighted_lightgbm": "Weighted LightGBM",
    "continuous_weighted_lightgbm": "连续权重 Weighted LightGBM",
    "catboost": "CatBoost baseline",
    "xgboost": "XGBoost baseline",
    "tree_blend": "Tree Blend",
}
PREDICTION_FILES = {
    "ridge": "official_baseline_valid_predictions.csv",
    "lightgbm": "official_lgbm_valid_predictions.csv",
    "weighted_lightgbm": "extreme_volatility_weighted_lgbm_valid_predictions.csv",
    "continuous_weighted_lightgbm": "extreme_volatility_continuous_weighted_lgbm_valid_predictions.csv",
    "catboost": "tree_baseline_catboost_valid_predictions.csv",
    "xgboost": "tree_baseline_xgboost_valid_predictions.csv",
}
TREE_BLEND_MODELS = ["lightgbm", "catboost", "xgboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Tree Blend 并对比整体与异常波动组指标")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--blend-objective", choices=["extreme-rmse"], default="extreme-rmse", help="融合权重搜索目标")
    parser.add_argument("--weight-step", type=float, default=0.05, help="融合权重网格步长")
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_prediction_frames(output_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []
    for model_name, file_name in PREDICTION_FILES.items():
        path = output_dir / file_name
        if not path.is_file():
            missing_files.append(str(path))
            continue
        frames[model_name] = pd.read_csv(path)
    if missing_files:
        joined = "\n".join(missing_files)
        raise FileNotFoundError(f"缺少以下预测文件，无法构建 Tree Blend:\n{joined}")
    return frames


def validate_prediction_alignment(frames: dict[str, pd.DataFrame]) -> np.ndarray:
    reference_name = next(iter(frames))
    reference = frames[reference_name]
    y_true = reference["y_true"].to_numpy(dtype=np.float64)
    for model_name, frame in frames.items():
        if len(frame) != len(reference):
            raise ValueError(f"{model_name} 验证预测行数不一致")
        if not frame["row_index"].equals(reference["row_index"]):
            raise ValueError(f"{model_name} 验证预测 row_index 不一致")
        current_y = frame["y_true"].to_numpy(dtype=np.float64)
        if not np.allclose(current_y, y_true, rtol=0, atol=1e-12):
            raise ValueError(f"{model_name} 验证预测 y_true 不一致")
    return y_true


def build_group_masks(y_true: np.ndarray, group_stats: pd.DataFrame) -> list[tuple[str, np.ndarray, str]]:
    q95 = float(group_stats["abs_label_q95_threshold"].iloc[0])
    abs_label = np.abs(y_true)
    return [
        (GROUP_OVERALL, np.ones(len(y_true), dtype=bool), "all validation samples"),
        (GROUP_EXTREME, abs_label > q95, f"abs(label) > {q95:.12g}"),
    ]


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return evaluate_regression(y_true[mask], y_pred[mask])


def validate_weight_step(weight_step: float) -> int:
    if weight_step <= 0 or weight_step > 1:
        raise ValueError("--weight-step 必须位于 (0, 1] 区间内")
    steps = int(round(1.0 / weight_step))
    if not np.isclose(steps * weight_step, 1.0, atol=1e-9):
        raise ValueError("--weight-step 必须能整除 1.0，例如 0.1、0.05、0.025")
    return steps


def search_blend_weights(
    y_true: np.ndarray,
    frames: dict[str, pd.DataFrame],
    extreme_mask: np.ndarray,
    *,
    weight_step: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    steps = validate_weight_step(weight_step)
    preds = {
        name: frames[name]["y_pred"].to_numpy(dtype=np.float64)
        for name in TREE_BLEND_MODELS
    }
    rows: list[dict[str, float]] = []
    for lgbm_units in range(steps + 1):
        for cat_units in range(steps + 1 - lgbm_units):
            xgb_units = steps - lgbm_units - cat_units
            weights = {
                "lightgbm": lgbm_units / steps,
                "catboost": cat_units / steps,
                "xgboost": xgb_units / steps,
            }
            blend_pred = (
                weights["lightgbm"] * preds["lightgbm"]
                + weights["catboost"] * preds["catboost"]
                + weights["xgboost"] * preds["xgboost"]
            )
            overall = evaluate_regression(y_true, blend_pred)
            extreme = metric_values(y_true, blend_pred, extreme_mask)
            rows.append(
                {
                    "w_lgbm": weights["lightgbm"],
                    "w_catboost": weights["catboost"],
                    "w_xgboost": weights["xgboost"],
                    "weight_sum": weights["lightgbm"] + weights["catboost"] + weights["xgboost"],
                    "overall_pearson": overall["pearson"],
                    "overall_rmse": overall["rmse"],
                    "overall_mae": overall["mae"],
                    "extreme_pearson": extreme["pearson"],
                    "extreme_rmse": extreme["rmse"],
                    "extreme_mae": extreme["mae"],
                }
            )
    search_result = pd.DataFrame(rows).sort_values(
        ["extreme_rmse", "overall_pearson", "overall_rmse"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    best = search_result.iloc[0]
    best_weights = {
        "lightgbm": float(best["w_lgbm"]),
        "catboost": float(best["w_catboost"]),
        "xgboost": float(best["w_xgboost"]),
    }
    return best_weights, search_result


def make_tree_blend_prediction(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    reference = frames["lightgbm"]
    y_pred = sum(
        weights[model_name] * frames[model_name]["y_pred"].to_numpy(dtype=np.float64)
        for model_name in TREE_BLEND_MODELS
    )
    prediction = pd.DataFrame(
        {
            "row_index": reference["row_index"],
            "y_true": reference["y_true"],
            "y_pred": y_pred,
            "model": "tree_blend",
        }
    )
    prediction.to_csv(output_dir / "tree_blend_valid_predictions.csv", index=False)
    return prediction


def compute_metrics(frames: dict[str, pd.DataFrame], group_stats: pd.DataFrame) -> pd.DataFrame:
    y_true = validate_prediction_alignment(frames)
    groups = build_group_masks(y_true, group_stats)
    rows: list[dict[str, object]] = []
    for model_name, frame in frames.items():
        y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
        residual = y_true - y_pred
        squared_error = residual**2
        absolute_error = np.abs(residual)
        total_squared_error = float(squared_error.sum())
        total_absolute_error = float(absolute_error.sum())
        for group_name, mask, definition in groups:
            metrics = evaluate_regression(y_true[mask], y_pred[mask])
            squared_error_sum = float(squared_error[mask].sum())
            absolute_error_sum = float(absolute_error[mask].sum())
            rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "group": group_name,
                    "definition": definition,
                    "sample_count": int(mask.sum()),
                    "sample_ratio": float(mask.mean()),
                    "pearson": metrics["pearson"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "squared_error_contribution": squared_error_sum / total_squared_error,
                    "absolute_error_contribution": absolute_error_sum / total_absolute_error,
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "model_label",
        "group",
        "sample_ratio",
        "pearson",
        "rmse",
        "mae",
        "squared_error_contribution",
        "absolute_error_contribution",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in frame[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_label"]),
                    str(row["group"]),
                    f"{row['sample_ratio']:.2%}",
                    f"{row['pearson']:.6f}",
                    f"{row['rmse']:.6f}",
                    f"{row['mae']:.6f}",
                    f"{row['squared_error_contribution']:.2%}",
                    f"{row['absolute_error_contribution']:.2%}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def save_figures(metrics: pd.DataFrame, weights: dict[str, float], figure_dir: Path) -> list[str]:
    ensure_dir(figure_dir)
    setup_style()
    saved: list[str] = []
    model_order = list(MODEL_LABELS.values())

    for group_name, file_name, title in [
        (GROUP_OVERALL, "tree_blend_overall_metrics.png", "整体指标对比"),
        (GROUP_EXTREME, "tree_blend_extreme_metrics.png", "异常波动组指标对比"),
    ]:
        plot_data = metrics[metrics["group"] == group_name].copy()
        plot_data["model_label"] = pd.Categorical(plot_data["model_label"], categories=model_order, ordered=True)
        long_data = plot_data.melt(
            id_vars=["model_label"],
            value_vars=["pearson", "rmse", "mae"],
            var_name="metric",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=long_data, x="model_label", y="value", hue="metric", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("模型")
        ax.set_ylabel("指标值")
        ax.legend(title="指标")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        path = figure_dir / file_name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    weight_data = pd.DataFrame(
        [
            {"model_label": MODEL_LABELS[model_name], "weight": weight}
            for model_name, weight in weights.items()
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=weight_data, x="model_label", y="weight", ax=ax, color="#4C72B0")
    ax.set_title("Tree Blend 最优权重")
    ax.set_xlabel("基础树模型")
    ax.set_ylabel("权重")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    path = figure_dir / "tree_blend_weights.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))
    return saved


def write_report(
    report_path: Path,
    metrics: pd.DataFrame,
    weights: dict[str, float],
    search_result: pd.DataFrame,
    figure_paths: list[str],
    weight_step: float,
) -> None:
    ensure_dir(report_path.parent)
    best = search_result.iloc[0]
    overall = metrics[metrics["group"] == GROUP_OVERALL].copy()
    extreme = metrics[metrics["group"] == GROUP_EXTREME].copy()
    overall_best_pearson = overall.sort_values("pearson", ascending=False).iloc[0]
    overall_best_rmse = overall.sort_values("rmse", ascending=True).iloc[0]
    extreme_best_rmse = extreme.sort_values("rmse", ascending=True).iloc[0]
    extreme_best_pearson = extreme.sort_values("pearson", ascending=False).iloc[0]
    if np.isclose(weights["lightgbm"], 1.0) and np.isclose(weights["catboost"], 0.0) and np.isclose(weights["xgboost"], 0.0):
        blend_note = "按异常组 RMSE 优化时，最优 Tree Blend 退化为纯 LightGBM，说明当前非负加权融合未进一步降低异常组 RMSE。"
    else:
        blend_note = "Tree Blend 在当前搜索网格下采用非退化融合权重，融合结果可与单模型直接比较。"
    figure_lines = "\n".join(f"- `{path}`" for path in figure_paths)
    content = f"""# Tree Blend 融合模型对比报告

生成时间：{datetime.now().isoformat(timespec="seconds")}

## 实验设置

- 融合模型：`Tree Blend = LightGBM + CatBoost + XGBoost`。
- 权重搜索目标：最小化异常波动组 RMSE。
- 权重约束：三个权重非负，且权重和为 1。
- 搜索步长：`{weight_step}`。
- 异常组定义：验证集 `abs(label)` 大于 95% 分位数。

## 最优融合权重

| base_model | weight |
| --- | ---: |
| LightGBM | {weights["lightgbm"]:.2f} |
| CatBoost baseline | {weights["catboost"]:.2f} |
| XGBoost baseline | {weights["xgboost"]:.2f} |

权重搜索最优行：

| overall_pearson | overall_rmse | overall_mae | extreme_pearson | extreme_rmse | extreme_mae |
| ---: | ---: | ---: | ---: | ---: | ---: |
| {best["overall_pearson"]:.6f} | {best["overall_rmse"]:.6f} | {best["overall_mae"]:.6f} | {best["extreme_pearson"]:.6f} | {best["extreme_rmse"]:.6f} | {best["extreme_mae"]:.6f} |

## 全模型指标对比

{markdown_table(metrics)}

## 简要结论

- 整体 Pearson 最高模型：`{overall_best_pearson["model_label"]}`。
- 整体 RMSE 最低模型：`{overall_best_rmse["model_label"]}`。
- 异常组 RMSE 最低模型：`{extreme_best_rmse["model_label"]}`。
- 异常组 Pearson 最高模型：`{extreme_best_pearson["model_label"]}`。
- {blend_note}
- 本次融合权重直接在当前验证集上搜索，用于课程实验分析，不作为严格无偏泛化估计。

## 图表文件

{figure_lines}
"""
    report_path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "experiments")
    report_dir = ensure_dir(root / "outputs" / "reports")
    figure_dir = ensure_dir(root / "outputs" / "figures" / "extreme_volatility")

    frames = load_prediction_frames(output_dir)
    y_true = validate_prediction_alignment(frames)
    group_stats = pd.read_csv(output_dir / "extreme_volatility_group_stats.csv")
    groups = build_group_masks(y_true, group_stats)
    extreme_mask = next(mask for group_name, mask, _ in groups if group_name == GROUP_EXTREME)

    weights, search_result = search_blend_weights(
        y_true,
        frames,
        extreme_mask,
        weight_step=args.weight_step,
    )
    search_path = output_dir / "tree_blend_weight_search.csv"
    search_result.to_csv(search_path, index=False, encoding="utf-8-sig")

    blend_pred = make_tree_blend_prediction(output_dir, frames, weights)
    frames["tree_blend"] = blend_pred
    metrics = compute_metrics(frames, group_stats)

    metrics_path = output_dir / "tree_blend_all_models_overall_extreme_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    figure_paths = save_figures(metrics, weights, figure_dir)
    report_path = report_dir / "tree_blend_all_models_compare.md"
    write_report(report_path, metrics, weights, search_result, figure_paths, args.weight_step)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "blend_objective": args.blend_objective,
        "weight_step": args.weight_step,
        "best_weights": weights,
        "best_weight_sum": float(sum(weights.values())),
        "tree_blend_prediction_csv": str(output_dir / "tree_blend_valid_predictions.csv"),
        "weight_search_csv": str(search_path),
        "metrics_csv": str(metrics_path),
        "report": str(report_path),
        "figures": figure_paths,
        "best_search_row": search_result.iloc[0].to_dict(),
    }
    summary_path = output_dir / "tree_blend_all_models_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
