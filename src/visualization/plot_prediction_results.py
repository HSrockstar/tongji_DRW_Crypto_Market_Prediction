from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_ROOT = Path(r"E:\DRW")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_model_metrics(root: Path, output_dir: Path) -> None:
    path = root / "outputs" / "experiments" / "official_model_compare.csv"
    if not path.is_file():
        print(f"未找到模型对比结果，跳过: {path}")
        return
    data = pd.read_csv(path)
    metric_data = data.melt(
        id_vars=["model"],
        value_vars=["pearson", "rmse", "mae"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metric_data, x="metric", y="value", hue="model")
    plt.title("模型验证指标对比")
    plt.xlabel("指标")
    plt.ylabel("数值")
    plt.tight_layout()
    plt.savefig(output_dir / "model_metrics_compare.png", dpi=150)
    plt.close()


def load_valid_predictions(root: Path) -> pd.DataFrame:
    files = [
        root / "outputs" / "experiments" / "official_baseline_valid_predictions.csv",
        root / "outputs" / "experiments" / "official_lgbm_valid_predictions.csv",
    ]
    frames = [pd.read_csv(path) for path in files if path.is_file()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_valid_prediction_distribution(root: Path, output_dir: Path) -> None:
    data = load_valid_predictions(root)
    if data.empty:
        print("未找到验证集预测结果，跳过验证预测分布图")
        return

    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=data, x="y_true", label="真实值", common_norm=False)
    sns.kdeplot(data=data, x="y_pred", hue="model", common_norm=False)
    plt.title("验证集真实值与预测值分布")
    plt.xlabel("数值")
    plt.ylabel("密度")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_prediction_distribution.png", dpi=150)
    plt.close()

    scatter_data = data.sample(min(len(data), 5000), random_state=42)
    plt.figure(figsize=(7, 7))
    sns.scatterplot(data=scatter_data, x="y_true", y="y_pred", hue="model", alpha=0.35, s=12)
    plt.title("验证集预测散点图")
    plt.xlabel("真实 label")
    plt.ylabel("预测值")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_prediction_scatter.png", dpi=150)
    plt.close()


def plot_submission_distribution(root: Path, output_dir: Path, submission_path: str | None) -> None:
    if submission_path:
        path = Path(submission_path)
        if not path.is_absolute():
            path = root / path
    else:
        smoke_path = root / "outputs" / "submissions" / "smoke_submission.csv"
        full_path = root / "outputs" / "submissions" / "submission.csv"
        path = smoke_path if smoke_path.is_file() else full_path

    if not path.is_file():
        print(f"未找到提交文件，跳过提交预测分布图: {path}")
        return
    data = pd.read_csv(path)
    plt.figure(figsize=(9, 5))
    sns.histplot(data["prediction"], bins=80, kde=True)
    plt.title(f"提交预测分布: {path.name}")
    plt.xlabel("prediction")
    plt.ylabel("样本数")
    plt.tight_layout()
    plt.savefig(output_dir / "submission_prediction_distribution.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成官方预测任务结果可视化图表")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--submission", default=None, help="可选提交文件路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "figures" / "prediction_task")
    setup_style()
    plot_model_metrics(root, output_dir)
    plot_valid_prediction_distribution(root, output_dir)
    plot_submission_distribution(root, output_dir, args.submission)
    print(f"预测任务图表已保存: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

