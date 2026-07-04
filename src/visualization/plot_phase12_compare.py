from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_ROOT = Path(__file__).resolve().parents[2]

STAGE_LABELS = {
    "handoff_ridge": "接手 Ridge",
    "handoff_lightgbm": "接手 LGBM",
    "phase1a_ensemble_default_lgbm": "阶段1a 集成",
    "phase1b_tuned_lgbm": "阶段1b 调参 LGBM",
    "phase1_tuned_ensemble": "阶段1 再集成",
    "phase2_combo_lgbm": "阶段2 组合特征",
}

STAGE_ORDER = list(STAGE_LABELS.keys())

STAGE_COLORS = {
    "handoff_ridge": "#4C72B0",
    "handoff_lightgbm": "#8DA0CB",
    "phase1a_ensemble_default_lgbm": "#55A868",
    "phase1b_tuned_lgbm": "#C44E52",
    "phase1_tuned_ensemble": "#8172B3",
    "phase2_combo_lgbm": "#CCB974",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_phase12_frame(root: Path) -> pd.DataFrame:
    path = root / "outputs" / "experiments" / "phase12_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"未找到阶段对比表: {path}")
    data = pd.read_csv(path)
    key_map = {
        ("handoff", "ridge"): "handoff_ridge",
        ("handoff", "lightgbm"): "handoff_lightgbm",
        ("phase1a_ensemble_default_lgbm", "blend_w0.5"): "phase1a_ensemble_default_lgbm",
        ("phase1b_tuned_lgbm", "lightgbm"): "phase1b_tuned_lgbm",
        ("phase1_tuned_ensemble", "blend_w0.5"): "phase1_tuned_ensemble",
        ("phase2_combo_lgbm", "synthesized_98feat"): "phase2_combo_lgbm",
    }
    data["stage_key"] = data.apply(lambda row: key_map[(row["stage"], row["model"])], axis=1)
    data["label"] = data["stage_key"].map(STAGE_LABELS)
    data["color"] = data["stage_key"].map(STAGE_COLORS)
    return data.set_index("stage_key").loc[STAGE_ORDER].reset_index()


def plot_holdout_pearson(root: Path, output_dir: Path) -> Path:
    data = load_phase12_frame(root)
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(
        data["label"],
        data["holdout_pearson"],
        color=data["color"],
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.axhline(0.091906, color="#4C72B0", linestyle="--", linewidth=1.2, label="接手 Ridge (0.092)")
    ax.axhline(0.076856, color="#8DA0CB", linestyle=":", linewidth=1.2, label="接手 LGBM (0.077)")
    ax.set_ylabel("Holdout Pearson")
    ax.set_xlabel("")
    ax.set_title("阶段 1+2 Holdout Pearson 对比")
    ax.set_ylim(0.06, 0.11)
    ax.legend(loc="upper left", frameon=True)
    for bar, value in zip(bars, data["holdout_pearson"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out_path = output_dir / "phase12_holdout_pearson_compare.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_phase12_only(root: Path, output_dir: Path) -> Path:
    data = load_phase12_frame(root)
    phase_data = data[~data["stage_key"].str.startswith("handoff")].copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        phase_data["label"],
        phase_data["holdout_pearson"],
        color=phase_data["color"],
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.axhline(0.102359, color="#C44E52", linestyle="--", linewidth=1.2, label="最优: 调参 LGBM (0.102)")
    ax.set_ylabel("Holdout Pearson")
    ax.set_title("阶段 1+2 四步实验对比（不含接手基线）")
    ax.set_ylim(0.088, 0.106)
    ax.legend(loc="lower right", frameon=True)
    for bar, value in zip(bars, phase_data["holdout_pearson"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    out_path = output_dir / "phase12_steps_pearson_compare.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成阶段1+2 Holdout Pearson 对比图")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="项目根目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = ensure_dir(root / "outputs" / "figures" / "prediction_task")
    setup_style()
    full_path = plot_holdout_pearson(root, output_dir)
    steps_path = plot_phase12_only(root, output_dir)
    print(f"已保存: {full_path}")
    print(f"已保存: {steps_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
