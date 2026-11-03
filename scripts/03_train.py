"""Step 03 - train the dual encoder with InfoNCE (E1) and hardest-negative triplet loss (E2).

Same features, split, architecture, optimiser, batch size, epoch budget and
model-selection rule for both; only the loss differs. 3 seeds each.
The test set is NOT used here (only in step 06).

Outputs
  results/03_history_<loss>_seed<s>.csv   per-epoch losses + train-probe/val recall
  results/03_runs.csv                     one row per run (best epoch, time, VRAM, val + train-probe metrics)
  results/03_val_summary.csv              mean and std over seeds, per method
  results/figures/03_training_curves.png
  cache/checkpoints/<loss>_seed<s>.pt
Run from the repo root:  python scripts/03_train.py            (both losses)
                         python scripts/03_train.py --loss infonce
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: F401  (must come before torch - see src/__init__.py)
import pandas as pd
import torch

import config
from src.training import prepare_data, train_dual_encoder
from src.utils import get_device, run_plots, save_csv

METHOD_NAME = {"infonce": "E1 InfoNCE", "triplet": "E2 Triplet (hardest negative)"}
CHECKPOINTS = config.CACHE / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["infonce", "triplet", "both"], default="both")
    args = parser.parse_args()
    losses = ["infonce", "triplet"] if args.loss == "both" else [args.loss]

    device = get_device()
    data = prepare_data(device)
    print(f"Device: {device} | train images {len(data['splits']['train']['img'])}, "
          f"val {len(data['splits']['val']['img'])}", flush=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    runs, histories = [], {}
    for loss_name in losses:
        for seed in config.SEEDS:
            print(f"\n=== {METHOD_NAME[loss_name]} | seed {seed} ===", flush=True)
            model, history, best, info = train_dual_encoder(data, loss_name, seed, device)
            hist = pd.DataFrame(history)
            histories[(loss_name, seed)] = hist
            save_csv(hist, config.RESULTS / f"03_history_{loss_name}_seed{seed}.csv")
            torch.save(model.state_dict(), CHECKPOINTS / f"{loss_name}_seed{seed}.pt")

            row = {"method": METHOD_NAME[loss_name], "loss": loss_name, "seed": seed, **info}
            row.update({f"val_{k}": v for k, v in best["val"].items()})
            row.update({f"train_probe_{k}": v for k, v in best["train_probe"].items()})
            row["rsum_gap_train_minus_val"] = round(best["train_probe"]["rsum"] - best["val"]["rsum"], 2)
            runs.append(row)
            print(f"  best epoch {info['best_epoch']} / ran {info['epochs_run']} | val rsum {best['val']['rsum']} | "
                  f"val R@1 i2t {best['val']['i2t_R@1']} t2i {best['val']['t2i_R@1']} | "
                  f"{info['train_seconds']}s", flush=True)

    runs_df = pd.DataFrame(runs)
    out_runs = config.RESULTS / "03_runs.csv"
    if args.loss != "both" and out_runs.exists():          # keep the other method's rows
        old = pd.read_csv(out_runs)
        runs_df = pd.concat([old[old["loss"] != args.loss], runs_df], ignore_index=True)
    save_csv(runs_df, out_runs)

    keep = ["val_i2t_R@1", "val_i2t_R@5", "val_i2t_R@10", "val_t2i_R@1", "val_t2i_R@5", "val_t2i_R@10",
            "val_i2t_MRR", "val_t2i_MRR", "val_rsum", "train_probe_rsum", "rsum_gap_train_minus_val",
            "best_epoch", "train_seconds"]
    summary = runs_df.groupby("method")[keep].agg(["mean", "std"]).round(3)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.insert(0, "n_seeds", runs_df.groupby("method").size())
    save_csv(summary.reset_index(), config.RESULTS / "03_val_summary.csv")

    run_plots("03")   # separate process, see src/plots.py

    print("\nStep 03 done. Validation summary (mean over seeds):")
    for _, r in summary.reset_index().iterrows():
        print(f"  {r['method']}: val rsum {r['val_rsum_mean']} (sd {r['val_rsum_std']}), "
              f"gap train-val {r['rsum_gap_train_minus_val_mean']}")


if __name__ == "__main__":
    main()
