from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preprocessing.build_features import MARKET_FIELDS
from data_preprocessing.preprocess import DEFAULT_ROOT, TARGET_COL, ensure_dir, load_json, load_parquet_frame


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_label_distribution(data: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(data[TARGET_COL], bins=80, kde=True)
    plt.title("label 分布")
    plt.xlabel("label")
    plt.ylabel("样本数")
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution.png", dpi=150)
    plt.close()


def plot_market_features(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for index, column in enumerate(MARKET_FIELDS):
        sns.histplot(data[column], bins=60, ax=axes[index])
        axes[index].set_title(column)
        axes[index].set_xlabel(column)
        axes[index].set_ylabel("样本数")
    axes[-1].axis("off")
    fig.suptitle("公开市场特征分布", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "market_feature_distributions.png", dpi=150)
    plt.close()


def plot_missing_inf_report(root: Path, output_dir: Path) -> None:
    report_path = root / "outputs" / "experiments" / "missing_inf_report.json"
    if not report_path.is_file():
        print(f"未找到缺失值检查报告，跳过: {report_path}")
        return

    report = load_json(report_path)
    rows = []
    for file_name, info in report["files"].items():
        rows.append({"file": file_name, "type": "null", "count": info["total_null"]})
        rows.append({"file": file_name, "type": "NaN", "count": info["total_nan"]})
        rows.append({"file": file_name, "type": "inf/-inf", "count": info["total_inf"]})
    plot_data = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_data, x="file", y="count", hue="type")
    plt.title("缺失值与无穷值检查摘要")
    plt.xlabel("文件")
    plt.ylabel("数量")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_inf_summary.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成数据预处理相关可视化图表")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    parser.add_argument("--sample-rows", type=int, default=100000, help="读取前 N 行绘图")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "figures" / "data_preprocessing")
    setup_style()

    data = load_parquet_frame(root, "train.parquet", sample_rows=args.sample_rows, include_label=True)
    plot_label_distribution(data, output_dir)
    plot_market_features(data, output_dir)
    plot_missing_inf_report(root, output_dir)
    print(f"数据预处理图表已保存: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

