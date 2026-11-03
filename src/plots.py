"""All figures for steps 03-07. Reads only the CSV/JSON files in results/.

This module must NOT import torch. On Windows + Anaconda, PyTorch and
numpy/matplotlib each ship their own Intel OpenMP DLL (libiomp5md.dll); using
both in one process crashes with "OMP: Error #15". So figures are drawn in a
separate process (scripts/plot.py), called at the end of each step.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.utils import save_figure

METHOD_NAME = {"infonce": "E1 InfoNCE", "triplet": "E2 Triplet (hardest negative)"}
BLUE, ORANGE = "#4C72B0", "#DD8452"


def plot_03():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, loss_name in enumerate(["infonce", "triplet"]):
        files = [config.RESULTS / f"03_history_{loss_name}_seed{s}.csv" for s in config.SEEDS]
        for i, f in enumerate(p for p in files if p.exists()):
            h = pd.read_csv(f)
            first = i == 0
            axes[0, col].plot(h["epoch"], h["train_probe_loss"], color=BLUE, alpha=0.8,
                              label="train probe" if first else None)
            axes[0, col].plot(h["epoch"], h["val_loss"], color=ORANGE, alpha=0.8,
                              label="validation" if first else None)
            axes[1, col].plot(h["epoch"], h["train_probe_rsum"], color=BLUE, alpha=0.8,
                              label="train probe" if first else None)
            axes[1, col].plot(h["epoch"], h["val_rsum"], color=ORANGE, alpha=0.8,
                              label="validation" if first else None)
            axes[1, col].axvline(h.loc[h["val_rsum"].idxmax(), "epoch"], color="grey", linestyle="--", linewidth=0.8)
        axes[0, col].set(title=f"{METHOD_NAME[loss_name]}: loss", xlabel="epoch", ylabel="loss (500 images)")
        axes[1, col].set(title=f"{METHOD_NAME[loss_name]}: R-sum (dashed = best epoch)", xlabel="epoch",
                         ylabel="R-sum (sum of 6 recalls, %)")
        for ax in axes[:, col]:
            if ax.lines:
                ax.legend()
    fig.suptitle("Training curves, 3 seeds per method: train probe vs validation (500 images each)")
    fig.tight_layout()
    save_figure(fig, config.FIGURES / "03_training_curves.png")
    plt.close(fig)


def plot_04():
    runs = pd.read_csv(config.RESULTS / "04_rerank_runs.csv")
    base = runs[runs["method"].str.startswith("E1")].set_index("seed")["val_rsum"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, seed in enumerate(config.SEEDS):
        f = config.RESULTS / f"04_rerank_history_seed{seed}.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        first = i == 0
        axes[0].plot(h["epoch"], h["val_rsum"], color=ORANGE, alpha=0.8,
                     label="E3 re-ranked (validation)" if first else None)
        axes[0].axhline(base[seed], color=BLUE, linestyle="--", linewidth=1,
                        label="E1 alone (validation)" if first else None)
        axes[1].plot(h["epoch"], h["train_bce_all_pairs"], color=BLUE, alpha=0.8, label="train" if first else None)
        axes[1].plot(h["epoch"], h["val_bce_all_pairs"], color=ORANGE, alpha=0.8, label="validation" if first else None)
    axes[0].set(title="(a) Validation R-sum: E1 vs E1 + re-ranker", xlabel="re-ranker epoch", ylabel="R-sum (%)")
    axes[1].set(title="(b) Re-ranker loss on true + hard-negative pairs", xlabel="epoch",
                ylabel="binary cross-entropy")
    for ax in axes:
        ax.legend()
    fig.suptitle("E3 concatenation + MLP re-ranker, 3 seeds")
    fig.tight_layout()
    save_figure(fig, config.FIGURES / "04_rerank_curves.png")
    plt.close(fig)


def plot_06():
    import json
    import numpy as np
    s = pd.read_csv(config.RESULTS / "06_test_summary.csv")
    rb = json.load(open(config.RESULTS / "06_random_baseline.json", encoding="utf-8"))["recall_percent"]
    colors = {"E1": BLUE, "E2": ORANGE, "E3": "#55A868", "E4": "#8172B2"}
    labels = {"E1": "E1 InfoNCE", "E2": "E2 Triplet", "E3": "E3 E1 + MLP re-ranker", "E4": "E4 CLIP zero-shot (ref.)"}
    ks = ["R@1", "R@5", "R@10"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (direction, title) in zip(axes, (("i2t", "(a) Image → text"), ("t2i", "(b) Text → image"))):
        x = np.arange(len(ks))
        width = 0.2
        for i, (_, r) in enumerate(s.iterrows()):
            means = [r[f"{direction}_{k}_mean"] for k in ks]
            stds = [0 if pd.isna(r[f"{direction}_{k}_std"]) else r[f"{direction}_{k}_std"] for k in ks]
            bars = ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3,
                          color=colors[r["code"]], label=labels[r["code"]])
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        ax.plot(x, [rb[f"{direction}_{k}"] for k in ks], "k_", markersize=25, label="random ranking")
        ax.axhline(10, color="grey", linestyle=":", linewidth=1)
        ax.axhline(25, color="grey", linestyle="--", linewidth=1)
        ax.set(title=title, xticks=x, xticklabels=ks, ylabel="recall on 500-image test set (%)", ylim=(0, 105))
    axes[0].text(2.45, 11, "target R@1", fontsize=7, color="grey", ha="right")
    axes[0].text(2.45, 26, "target R@5", fontsize=7, color="grey", ha="right")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Test retrieval (mean ± sd over 3 seeds; E4 single deterministic run)")
    fig.tight_layout()
    save_figure(fig, config.FIGURES / "06_test_recall.png")
    plt.close(fig)


def plot_07():
    import numpy as np
    colors = {"E1": BLUE, "E2": ORANGE, "E3": "#55A868", "E4": "#8172B2"}
    labels = {"E1": "E1 InfoNCE", "E2": "E2 Triplet", "E3": "E3 re-ranker", "E4": "E4 CLIP (ref.)"}
    audit = pd.read_csv(config.RESULTS / "07_gender_audit.csv")
    scenes = pd.read_csv(config.RESULTS / "07_scene_groups.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})

    groups = ["male terms", "female terms", "both", "neither"]
    x = np.arange(len(groups))
    for i, code in enumerate(["E1", "E2", "E3", "E4"]):
        a = audit[audit["code"] == code].set_index("gender_group").reindex(groups)
        axes[0].bar(x + (i - 1.5) * 0.2, a["R@5"], 0.2, color=colors[code], label=labels[code])
    n = audit[audit["code"] == "E1"].set_index("gender_group").reindex(groups)["n_queries"]
    axes[0].set(title="(a) Text → image R@5 by gendered words in the query", ylabel="R@5 on test (%)",
                xticks=x, xticklabels=[f"{g}\n(n={int(c)})" for g, c in zip(groups, n)], ylim=(0, 100))
    axes[0].legend(fontsize=8, loc="lower right")

    names = list(dict.fromkeys(scenes["scene_group"]))
    x = np.arange(len(names))
    for i, code in enumerate(["E1", "E2", "E3", "E4"]):
        s = scenes[scenes["code"] == code].set_index("scene_group").reindex(names)
        axes[1].bar(x + (i - 1.5) * 0.2, s["R@5"], 0.2, color=colors[code], label=labels[code])
    s1 = scenes[scenes["code"] == "E1"].set_index("scene_group").reindex(names)
    axes[1].set(title="(b) Text → image R@5 by scene keyword group", ylabel="R@5 on test (%)", xticks=x,
                xticklabels=[f"{g}\n(test n={int(t)}; {p:.1f}% of train)" for g, t, p in
                             zip(names, s1["n_test_queries"], s1["share_of_train_captions_pct"])], ylim=(0, 100))
    axes[1].tick_params(axis="x", labelsize=8)
    fig.suptitle("Bias and scene-group audit on the test set (E1-E3: mean over 3 seeds)")
    fig.tight_layout()
    save_figure(fig, config.FIGURES / "07_bias_and_groups.png")
    plt.close(fig)


PLOTS = {"03": plot_03, "04": plot_04, "06": plot_06, "07": plot_07}
