"""Create publication-ready visualizations for the FewShot experiment log.

Usage:
    python visualize_results.py --input results_data.txt --output-dir .

The script parses repeated ``key: value`` experiment blocks, writes a tidy CSV,
and produces four figures plus a short Markdown report. The input must contain
one curated, valid run per method; duplicate methods require explicit curation.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


COLORS = {
    "ResNet baselines": "#7A5195",
    "Prototypical networks": "#00A6A6",
    "CLIP without parameter training": "#F28E2B",
    "CLIP with parameter training": "#4E79A7",
}


def parse_scalar(value: str):
    """Convert percentages and simple numeric strings while preserving text."""
    value = value.strip()
    candidate = value[:-1] if value.endswith("%") else value
    if re.fullmatch(r"-?\d+", candidate):
        return int(candidate)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", candidate):
        return float(candidate)
    return value


def parse_blocks(path: Path) -> list[dict]:
    """Parse section headings followed by key-value pairs from the text log."""
    blocks: list[dict] = []
    current: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            if any(key != "Section" for key in current):
                blocks.append(current)
                current = {}
            continue
        if set(line) == {"="}:
            continue
        if ":" not in line:
            current["Section"] = line
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = parse_scalar(value)
    if any(key != "Section" for key in current):
        blocks.append(current)
    return blocks


def display_name(block: dict) -> str:
    experiment = str(block.get("Experiment", ""))
    section = str(block.get("Section", ""))
    names = {
        "data_baseline_pretrained_resnet18": "ResNet18 full fine-tune",
        "data_baseline_frozen_pretrained_resnet18": "ResNet18 linear probe",
        "model_baseline_frozen_pretrained_protonet": "ProtoNet, frozen ResNet18",
        "from_scratch_protonet": "ProtoNet, from scratch",
        "clip_zero_shot": "CLIP zero-shot",
        "clip_linear_probe": "CLIP linear probe",
        "clip_hard_prompt_zero_shot": "CLIP hard prompt",
        "clip_coop": "CLIP CoOp",
        "clip_tip_adapter": "CLIP Tip-Adapter",
        "clip_lora": "CLIP LoRA",
        "clip_adapter": "CLIP-Adapter",
    }
    if experiment in names:
        return names[experiment]
    if section == "White-Box Baseline":
        return "Random ResNet18 baseline"
    return experiment or section or str(block.get("Model", "Unknown"))


def method_family(name: str) -> str:
    if name.startswith("CLIP"):
        return ("CLIP without parameter training"
                if name in {"CLIP zero-shot", "CLIP hard prompt", "CLIP Tip-Adapter"}
                else "CLIP with parameter training")
    if name.startswith("ProtoNet"):
        return "Prototypical networks"
    return "ResNet baselines"


def to_dataframe(blocks: list[dict]) -> pd.DataFrame:
    rows = []
    for block in blocks:
        method = display_name(block)
        validation = block.get("Best Validation Accuracy", block.get("Validation Accuracy"))
        test = block.get("Test Accuracy")
        if validation is None or test is None:
            continue
        trainable_parameters = block.get("Trainable Parameters", block.get("Trainable LoRA Parameters"))
        parameter_training = method not in {
            "CLIP zero-shot", "CLIP hard prompt", "CLIP Tip-Adapter",
            "ProtoNet, frozen ResNet18",
        }
        rows.append(
            {
                "method": method,
                "experiment": block.get("Experiment", "white_box_baseline"),
                "family": method_family(method),
                "model": block.get("Model", block.get("Backbone", "")),
                "validation_accuracy": float(validation),
                "test_accuracy": float(test),
                "test_minus_validation_pp": float(test) - float(validation),
                "best_epoch": block.get("Best Epoch"),
                "epochs": block.get("Epochs", block.get("Training Epochs")),
                "k_shot": block.get("K-shot", 0),
                "episode_n_shot": block.get("N-shot"),
                "seed": block.get("Seed"),
                "run_id": block.get("Run ID"),
                "trainable_parameters": trainable_parameters,
                "trainable_ratio_percent": block.get("Trainable Ratio"),
                "parameter_training": parameter_training,
                "training_mode": "trained" if parameter_training else "no parameter training",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No experiment blocks with validation and test accuracy were found.")
    if frame["method"].duplicated().any():
        duplicates = frame.loc[frame["method"].duplicated(), "method"].tolist()
        raise ValueError(f"Duplicate methods: {duplicates}. Curate valid runs before plotting.")
    return frame


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B8B8B8",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )


def finish_figure(fig: plt.Figure, output_dir: Path, stem: str, pdf: PdfPages) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ranking(df: pd.DataFrame, output_dir: Path, pdf: PdfPages) -> None:
    ranked = df.sort_values("test_accuracy", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    y = np.arange(len(ranked))
    bars = ax.barh(y, ranked["test_accuracy"], color=[COLORS[v] for v in ranked["family"]], height=0.68)
    ax.set_yticks(y, ranked["method"])
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("FewShot experiment ranking", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.text(0, 1.01, "Mini-ImageNet, one reported run per method", transform=ax.transAxes, color="#666666")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, ranked["test_accuracy"]):
        ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontweight="bold", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=color, markersize=8, label=family) for family, color in COLORS.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2)
    finish_figure(fig, output_dir, "01_test_accuracy_ranking", pdf)


def plot_val_test_gap(df: pd.DataFrame, output_dir: Path, pdf: PdfPages) -> None:
    ordered = df.sort_values("test_accuracy", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    y = np.arange(len(ordered))
    for i, row in ordered.iterrows():
        low, high = sorted([row["validation_accuracy"], row["test_accuracy"]])
        ax.plot([low, high], [i, i], color="#B8B8B8", linewidth=2.2, zorder=1)
    ax.scatter(ordered["validation_accuracy"], y, s=55, color="#F28E2B", marker="o", label="Validation", zorder=3)
    ax.scatter(ordered["test_accuracy"], y, s=60, color="#4E79A7", marker="D", label="Test", zorder=3)
    ax.set_yticks(y, ordered["method"])
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Validation and test accuracy", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.text(0, 1.01, "Lines connect validation and test accuracy for the same method", transform=ax.transAxes, color="#666666")
    left = min(ordered["validation_accuracy"].min(), ordered["test_accuracy"].min()) - 4
    right = max(ordered["validation_accuracy"].max(), ordered["test_accuracy"].max()) + 4
    ax.set_xlim(left, right)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axvline(ordered["test_accuracy"].mean(), color="#888888", linestyle="--", linewidth=1, alpha=0.8)
    ax.text(ordered["test_accuracy"].mean() + 0.3, len(ordered) - 0.35, f"mean test {ordered['test_accuracy'].mean():.1f}%", color="#666666", fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    finish_figure(fig, output_dir, "02_validation_test_gap", pdf)


def plot_parameter_efficiency(df: pd.DataFrame, output_dir: Path, pdf: PdfPages) -> None:
    subset = df[df["method"].str.startswith("CLIP") & df["trainable_parameters"].notna()].copy()
    subset["trainable_parameters"] = pd.to_numeric(subset["trainable_parameters"])
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.scatter(subset["trainable_parameters"], subset["test_accuracy"], s=130, color=COLORS["CLIP with parameter training"], edgecolor="white", linewidth=1.2)
    for _, row in subset.iterrows():
        ax.annotate(
            f"{row['method']}\n{int(row['trainable_parameters']):,} params",
            (row["trainable_parameters"], row["test_accuracy"]),
            xytext=(8, -32) if row["method"] == "CLIP-Adapter" else (8, 7),
            textcoords="offset points", fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlim(subset["trainable_parameters"].min() / 2, subset["trainable_parameters"].max() * 6)
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Parameter-efficient CLIP adaptation", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.text(0, 1.01, "Methods with reported trainable-parameter counts", transform=ax.transAxes, color="#666666")
    ax.set_ylim(subset["test_accuracy"].min() - 3, subset["test_accuracy"].max() + 3)
    ax.grid(color="#DDDDDD", linewidth=0.8, which="both")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, output_dir, "03_clip_parameter_efficiency", pdf)


def plot_clip_comparison(df: pd.DataFrame, output_dir: Path, pdf: PdfPages) -> None:
    subset = df[df["method"].str.startswith("CLIP")].sort_values("test_accuracy")
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    y = np.arange(len(subset))
    bars = ax.barh(y, subset["test_accuracy"], color=[COLORS[f] for f in subset["family"]])
    ax.set_yticks(y, subset["method"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("CLIP adaptation: parameter training vs. training-free methods",
                 loc="left", fontsize=13, fontweight="bold", pad=15)
    for bar, value in zip(bars, subset["test_accuracy"]):
        ax.text(value + .6, bar.get_y() + bar.get_height()/2, f"{value:.2f}%", va="center")
    ax.grid(axis="x", color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=COLORS[name], label=name)
               for name in ["CLIP with parameter training", "CLIP without parameter training"]]
    ax.set_ylim(-1.7, len(subset) - .4)
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    fig.text(.13, .005, "Hard prompt: WordNet descriptions. Tip-Adapter: labeled support-feature cache. Seed 42 only.",
             fontsize=9, color="#666666")
    finish_figure(fig, output_dir, "04_clip_adaptation_comparison", pdf)


def write_report(df: pd.DataFrame, output_dir: Path) -> None:
    best = df.loc[df["test_accuracy"].idxmax()]
    clip_zero = df.loc[df["method"] == "CLIP zero-shot", "test_accuracy"].iloc[0]
    strongest_non_clip = df.loc[~df["method"].str.startswith("CLIP")].sort_values("test_accuracy", ascending=False).iloc[0]
    gap = best["test_accuracy"] - clip_zero
    trained = df[df["family"] == "CLIP with parameter training"].sort_values("test_accuracy", ascending=False)
    training_free = df[df["method"].isin(["CLIP hard prompt", "CLIP Tip-Adapter"])].sort_values("test_accuracy", ascending=False)
    comparison = "\n".join(f"| {r.method} | {r.test_accuracy:.2f}% |" for r in pd.concat([trained, training_free]).itertuples())
    report = f"""# FewShot experiment visualization summary

Source: `results_data.txt`
Experiments parsed: **{len(df)}**

## Main observations

1. **{best['method']}** achieves the highest reported test accuracy at **{best['test_accuracy']:.2f}%**.
2. It improves on CLIP zero-shot by **{gap:.2f} percentage points**.
3. The strongest non-CLIP result is **{strongest_non_clip['method']}** at **{strongest_non_clip['test_accuracy']:.2f}%**.
4. Validation and test accuracy are close for most methods, but these are single-seed results and do not quantify run-to-run uncertainty.

## Parameter training and training-free CLIP adaptation

| Method | Test accuracy |
| --- | ---: |
{comparison}

All four tested CLIP methods with parameter training outperform both Hard Prompt and the training-free Tip-Adapter in this run. Their accuracies span **{trained['test_accuracy'].min():.2f}%–{trained['test_accuracy'].max():.2f}%**.
Hard Prompt adds WordNet descriptions without labeled support images. Tip-Adapter uses a labeled image-feature cache (alpha=1, beta=5), not an external semantic knowledge base. It has hyperparameters but no gradient-based parameter training in this implementation.
These comparisons describe this dataset, configuration and seed; they do not establish universal superiority or statistical significance. LoRA and CLIP-Adapter differ by only 0.18 percentage points.

## Generated files

- `01_test_accuracy_ranking.png`: overall comparison and method families.
- `02_validation_test_gap.png`: validation/test gap for each experiment.
- `03_clip_parameter_efficiency.png`: accuracy against reported trainable parameters.
- `04_clip_adaptation_comparison.png`: CLIP parameter-training versus training-free results.
- `fewshot_visualizations.pdf`: all figures in one multi-page PDF.
- `experiment_summary.csv`: parsed, analysis-ready experiment table.
- `visualize_results.py`: reproducible visualization code.

## Interpretation note

The parameter-efficiency figure includes only methods whose logs report trainable-parameter counts. The data contains one seed (`42`), so confidence intervals and significance claims are not supported.
"""
    (output_dir / "visualization_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("results_data.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    frame = to_dataframe(parse_blocks(args.input))
    frame.to_csv(args.output_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    with PdfPages(args.output_dir / "fewshot_visualizations.pdf") as pdf:
        plot_ranking(frame, args.output_dir, pdf)
        plot_val_test_gap(frame, args.output_dir, pdf)
        plot_parameter_efficiency(frame, args.output_dir, pdf)
        plot_clip_comparison(frame, args.output_dir, pdf)
    write_report(frame, args.output_dir)
    print(frame[["method", "validation_accuracy", "test_accuracy", "test_minus_validation_pp"]].sort_values("test_accuracy", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
