"""Generate publication-quality figures for the paper."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def plot_main_comparison(results_path: str, output_dir: str):
    """Bar chart comparing WindowProof vs baselines."""
    with open(results_path) as f:
        results = json.load(f)

    methods = []
    f1_scores = []
    auroc_scores = []

    # WindowProof (use weighted F1 for fair comparison)
    methods.append("WindowProof\n(Ours)")
    f1_scores.append(results["windowproof"]["weighted_f1"])
    auroc_scores.append(None)  # three-way, no single AUROC

    for key, label in [
        ("baseline_if", "IF\n(Full Features)"),
        ("baseline_lof", "LOF\n(Full Features)"),
        ("baseline_sparse_if", "IF\n(Sparse Only)"),
        ("baseline_sketch_if", "IF\n(Sketch Only)"),
    ]:
        if key in results:
            methods.append(label)
            f1_scores.append(results[key]["f1"])
            auroc_scores.append(results[key].get("auroc"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(methods))
    width = 0.6

    colors = ["#2196F3"] + ["#90CAF9"] * (len(methods) - 1)
    bars = ax.bar(x, f1_scores, width, color=colors, edgecolor="black", linewidth=0.5)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Method")
    ax.set_ylabel("F1 Score")
    ax.set_title("Anomaly Detection Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(os.path.join(output_dir, "main_comparison.pdf"))
    fig.savefig(os.path.join(output_dir, "main_comparison.png"))
    plt.close(fig)
    print("  Saved main_comparison.pdf/png")


def plot_ablation(ablation_path: str, output_dir: str):
    """Ablation study results."""
    with open(ablation_path) as f:
        results = json.load(f)

    labels_map = {
        "full_model": "Full Model",
        "no_sketch_residual": "w/o Sketch\nResidual",
        "no_integrity_flags": "w/o Integrity\nFlags",
        "no_committed_sketch": "w/o Committed\nSketch",
        "checkpoint_only": "Checkpoint\nOnly",
        "sketch_only": "Sketch\nOnly",
    }

    names = []
    f1s = []
    aurocs = []
    for key in ["full_model", "no_sketch_residual", "no_integrity_flags",
                "no_committed_sketch", "checkpoint_only", "sketch_only"]:
        if key in results:
            names.append(labels_map.get(key, key))
            f1s.append(results[key]["f1"])
            aurocs.append(results[key].get("auroc", 0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, f1s, width, label="F1", color="#2196F3",
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, aurocs, width, label="AUROC", color="#FF9800",
                   edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars1, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(os.path.join(output_dir, "ablation_study.pdf"))
    fig.savefig(os.path.join(output_dir, "ablation_study.png"))
    plt.close(fig)
    print("  Saved ablation_study.pdf/png")


def plot_attack_coverage(results_path: str, output_dir: str):
    """Heatmap of attack type detection rates."""
    with open(results_path) as f:
        results = json.load(f)

    coverage = results.get("attack_coverage", {})
    if not coverage or all(k == "none" for k in coverage):
        print("  No attack coverage data to plot.")
        return

    attack_types = [k for k in coverage if k != "none"]
    if not attack_types:
        return

    categories = {
        "Behavioral": ["detour", "loop", "abnormal_stop", "speed_burst", "teleport"],
        "Integrity": ["point_deletion", "point_injection", "timestamp_shift", "replay"],
    }

    data = []
    labels = []
    for cat, types in categories.items():
        for t in types:
            if t in coverage:
                data.append(coverage[t]["detection_rate"])
                labels.append(t.replace("_", "\n"))
            else:
                data.append(0)
                labels.append(t.replace("_", "\n"))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = ["#4CAF50" if d > 0.7 else "#FF9800" if d > 0.4 else "#F44336" for d in data]
    bars = ax.barh(range(len(labels)), data, color=colors, edgecolor="black", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, data)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", ha="left", va="center", fontsize=9)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Detection Rate")
    ax.set_title("Attack Type Detection Coverage")
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    # Add category labels
    ax.axhline(y=4.5, color="black", linewidth=0.8)
    ax.text(-0.15, 2, "Behavioral", fontsize=10, fontweight="bold",
            ha="center", va="center", rotation=90, transform=ax.get_yaxis_transform())
    ax.text(-0.15, 6.5, "Integrity", fontsize=10, fontweight="bold",
            ha="center", va="center", rotation=90, transform=ax.get_yaxis_transform())

    fig.savefig(os.path.join(output_dir, "attack_coverage.pdf"))
    fig.savefig(os.path.join(output_dir, "attack_coverage.png"))
    plt.close(fig)
    print("  Saved attack_coverage.pdf/png")


def plot_density_sweep(sweep_path: str, output_dir: str):
    """Detection F1 vs blockchain cost at different checkpoint densities."""
    with open(sweep_path) as f:
        results = json.load(f)

    densities = sorted(results.keys(), key=float)
    f1_vals = [results[d]["f1"] for d in densities]
    storage = [results[d]["storage_bytes"] for d in densities]
    gas = [results[d]["gas_cost"] for d in densities]
    density_vals = [float(d) for d in densities]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(density_vals, f1_vals, "o-", color="#2196F3", linewidth=2, markersize=6)
    ax1.set_xlabel("Checkpoint Density")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("Detection Performance vs. Density")
    ax1.grid(alpha=0.3)

    ax2.plot(density_vals, [s / 1024 for s in storage], "s-", color="#FF9800",
             linewidth=2, markersize=6)
    ax2.set_xlabel("Checkpoint Density")
    ax2.set_ylabel("Storage (KB)")
    ax2.set_title("Blockchain Storage vs. Density")
    ax2.grid(alpha=0.3)

    fig.suptitle("Checkpoint Density Tradeoff Analysis", fontsize=12, y=1.02)
    fig.tight_layout()

    fig.savefig(os.path.join(output_dir, "density_sweep.pdf"))
    fig.savefig(os.path.join(output_dir, "density_sweep.png"))
    plt.close(fig)
    print("  Saved density_sweep.pdf/png")


def plot_confusion_matrix(results_path: str, output_dir: str):
    """Three-way confusion matrix for WindowProof."""
    with open(results_path) as f:
        results = json.load(f)

    cm = np.array(results["windowproof"]["confusion_matrix"])
    labels = ["Normal", "Integrity\nFailure", "Behavioral\nAnomaly"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("WindowProof Three-Way Confusion Matrix")

    fig.savefig(os.path.join(output_dir, "confusion_matrix.pdf"))
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig)
    print("  Saved confusion_matrix.pdf/png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results_file = os.path.join(args.results_dir, "results.json")
    ablation_file = os.path.join(args.results_dir, "ablation_results.json")
    sweep_file = os.path.join(args.results_dir, "density_sweep.json")

    print("Generating figures...")

    if os.path.exists(results_file):
        plot_main_comparison(results_file, args.output_dir)
        plot_attack_coverage(results_file, args.output_dir)
        plot_confusion_matrix(results_file, args.output_dir)

    if os.path.exists(ablation_file):
        plot_ablation(ablation_file, args.output_dir)

    if os.path.exists(sweep_file):
        plot_density_sweep(sweep_file, args.output_dir)

    print("Done!")
