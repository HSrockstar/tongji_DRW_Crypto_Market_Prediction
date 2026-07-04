from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PEARSON = 0.10235862829968674

STEP_LABELS = {
    "baseline": "基线 phase1b",
    "step1": "Step1 多seed",
    "step2": "Step2 Purged调参",
    "step3": "Step3 树模型集成",
    "step4": "Step4 时序特征",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_summary(root: Path) -> pd.DataFrame:
    path = root / "outputs" / "experiments" / "overnight" / "optimization_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"未找到汇总表: {path}")
    data = pd.read_csv(path)
    data["label"] = data.apply(
        lambda row: STEP_LABELS.get(str(row["step"]), str(row["experiment"])),
        axis=1,
    )
    return data


def plot_pearson_compare(data: pd.DataFrame, output_dir: Path) -> None:
    ordered = data.sort_values("holdout_pearson", ascending=True)
    colors = ["#C44E52" if bool(row["beats_baseline"]) else "#4C72B0" for _, row in ordered.iterrows()]
    if ordered.iloc[-1]["step"] == "baseline":
        colors[-1] = "#8172B3"

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(ordered["label"], ordered["holdout_pearson"], color=colors, edgecolor="#333333")
    ax.axvline(BASELINE_PEARSON, color="#C44E52", linestyle="--", linewidth=1.2, label=f"基线 {BASELINE_PEARSON:.3f}")
    ax.set_xlabel("Holdout Pearson")
    ax.set_title("夜间优化实验 Holdout Pearson 对比")
    ax.legend(loc="lower right")
    for bar, value in zip(bars, ordered["holdout_pearson"], strict=True):
        ax.text(value + 0.0005, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "overnight_pearson_compare.png", dpi=150)
    plt.close(fig)


def plot_metric_panel(data: pd.DataFrame, output_dir: Path) -> None:
    plot_data = data.melt(
        id_vars=["label", "step"],
        value_vars=["holdout_pearson", "holdout_rmse", "holdout_mae"],
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "holdout_pearson": "Pearson",
        "holdout_rmse": "RMSE",
        "holdout_mae": "MAE",
    }
    plot_data["metric"] = plot_data["metric"].map(metric_labels)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=plot_data, x="label", y="value", hue="metric")
    ax.set_title("夜间优化实验指标汇总")
    ax.set_xlabel("")
    ax.set_ylabel("数值")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "overnight_metrics_panel.png", dpi=150)
    plt.close(fig)


def plot_delta_baseline(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data[data["step"] != "baseline"].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#55A868" if value > 0 else "#DD8452" for value in subset["vs_baseline"]]
    ax.bar(subset["label"], subset["vs_baseline"], color=colors, edgecolor="#333333")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_ylabel("相对基线 ΔPearson")
    ax.set_title("各实验相对 phase1b 基线的提升")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "overnight_delta_baseline.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制夜间优化汇总图")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = root / "outputs" / "figures" / "prediction_task"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    data = load_summary(root)
    plot_pearson_compare(data, output_dir)
    plot_metric_panel(data, output_dir)
    plot_delta_baseline(data, output_dir)
    print(f"优化汇总图已保存: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
