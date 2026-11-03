"""Step 04 - E3: train the concatenation + MLP re-ranker on top of each E1 (InfoNCE) model.

For each of the 3 InfoNCE seeds: take E1's frozen embeddings, find hard negatives
inside E1's top-10, train MatchMLP to tell true pairs from them, and measure
validation retrieval after re-ordering the top-10. Test set not used here.

Outputs
  results/04_rerank_history_seed<s>.csv   per-epoch BCE (train/val) and val recall
  results/04_rerank_runs.csv              per seed: E1 alone vs E1 + re-ranker on validation
  results/04_val_summary.csv              mean and std over seeds
  results/figures/04_rerank_curves.png
  cache/checkpoints/rerank_seed<s>.pt
Run from the repo root:  python scripts/04_rerank.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: F401  (must come before torch - see src/__init__.py)
import pandas as pd
import torch

import config
from src.metrics import retrieval_metrics
from src.rerank import CHECKPOINTS, load_dual_encoder, train_reranker
from src.training import embed_split, prepare_data
from src.utils import get_device, peak_vram_gb, reset_peak_vram, run_plots, save_csv


def main():
    device = get_device()
    data = prepare_data(device)
    print(f"Device: {device}", flush=True)
    va_owner = data["splits"]["val"]["owner"]

    runs = []
    for seed in config.SEEDS:
        print(f"\n=== E3 re-ranker on E1 InfoNCE seed {seed} ===", flush=True)
        base = load_dual_encoder(data, "infonce", seed, device)
        va_img, va_txt = embed_split(base, data, "val")
        base_m = retrieval_metrics(va_img @ va_txt.T, va_owner)

        reset_peak_vram()
        reranker, history, best, info = train_reranker(data, base, seed, device)
        info["peak_vram_gb"] = peak_vram_gb()
        save_csv(pd.DataFrame(history), config.RESULTS / f"04_rerank_history_seed{seed}.csv")
        torch.save(reranker.state_dict(), CHECKPOINTS / f"rerank_seed{seed}.pt")

        for name, m in (("E1 InfoNCE (no re-ranking)", base_m), ("E3 E1 + concat-MLP re-ranker", best["val"])):
            runs.append({"method": name, "seed": seed, **({**info} if name.startswith("E3") else {}),
                         **{f"val_{k}": v for k, v in m.items()}})
        print(f"  best epoch {info['best_epoch']} | val rsum E1 {base_m['rsum']} -> E3 {best['val']['rsum']} | "
              f"R@1 i2t {base_m['i2t_R@1']} -> {best['val']['i2t_R@1']}, "
              f"t2i {base_m['t2i_R@1']} -> {best['val']['t2i_R@1']} | {info['train_seconds']}s", flush=True)

    runs_df = pd.DataFrame(runs)
    save_csv(runs_df, config.RESULTS / "04_rerank_runs.csv")
    keep = [c for c in runs_df.columns if c.startswith("val_") and not c.endswith("median_rank")]
    summary = runs_df.groupby("method", sort=False)[keep].agg(["mean", "std"]).round(3)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.insert(0, "n_seeds", runs_df.groupby("method", sort=False).size())
    save_csv(summary.reset_index(), config.RESULTS / "04_val_summary.csv")

    run_plots("04")   # separate process, see src/plots.py

    print("\nStep 04 done. Validation (mean over seeds):")
    for _, r in summary.reset_index().iterrows():
        print(f"  {r['method']}: rsum {r['val_rsum_mean']} | R@1 i2t {r['val_i2t_R@1_mean']} "
              f"t2i {r['val_t2i_R@1_mean']} | R@10 i2t {r['val_i2t_R@10_mean']} t2i {r['val_t2i_R@10_mean']}")


if __name__ == "__main__":
    main()
