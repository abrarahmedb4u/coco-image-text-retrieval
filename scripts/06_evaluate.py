"""Step 06 - final evaluation of every method on the held-out TEST set (500 images).

This is the only step that uses the test set. Nothing is tuned here.
  E0 random ranking (calculated)          E1 InfoNCE dual encoder (3 seeds)
  E2 triplet dual encoder (3 seeds)       E3 E1 + concat-MLP re-ranker (3 seeds)
  E4 zero-shot CLIP ViT-B/32 (reference, deterministic)

Outputs
  results/06_test_per_seed.csv     every metric, every run
  results/06_test_summary.csv      mean / std over seeds + proposal-target check
  results/06_random_baseline.json  chance level on this gallery
  results/06_overfitting.csv       train-probe vs validation vs test R-sum
  results/06_efficiency.csv        parameters, training time, GPU memory, ms per query
  results/06_test_ranks.csv        rank of the correct item for every test query (used by step 07)
  results/figures/06_test_recall.png
Run from the repo root:  python scripts/06_evaluate.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: F401  (must come before torch - see src/__init__.py)
import pandas as pd
import torch

import config
from src.features import CLIP_FILE
from src.metrics import metrics_from_ranks, random_baseline, ranks_from_scores
from src.models import MatchMLP
from src.rerank import CHECKPOINTS, load_dual_encoder, rerank_ranks
from src.training import embed_split, prepare_data
from src.utils import get_device, load_json, run_plots, save_csv, save_json

TARGET_R1, TARGET_R5 = 10.0, 25.0   # proposal Section 1, both directions
NAMES = {"E1": "E1 InfoNCE", "E2": "E2 Triplet (hardest negative)",
         "E3": "E3 E1 + concat-MLP re-ranker", "E4": "E4 CLIP ViT-B/32 zero-shot (reference)"}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def query_latency_ms(device, query_fn, n_queries):
    """Mean wall-clock ms to answer one query at a time (gallery embeddings precomputed)."""
    for q in range(min(20, n_queries)):   # warm-up
        query_fn(q)
    sync(device)
    start = time.perf_counter()
    for q in range(n_queries):
        query_fn(q)
    sync(device)
    return round(1000 * (time.perf_counter() - start) / n_queries, 4)


def main():
    device = get_device()
    data = prepare_data(device)
    test = data["splits"]["test"]
    owner = test["owner"]
    image_ids = [data["image_ids"][int(p)] for p in test["img"]]
    caption_ids = test["cap"].tolist()   # caption_id == row position in features (see step 02)
    k = config.RERANK_K
    print(f"Device: {device} | test: {len(image_ids)} images, {len(caption_ids)} captions", flush=True)

    per_seed, rank_rows, latency = [], [], {}

    def record(code, seed, i2t, t2i):
        per_seed.append({"code": code, "method": NAMES[code], "seed": seed, **metrics_from_ranks(i2t, t2i)})
        rank_rows.extend({"code": code, "seed": seed, "direction": "i2t", "query_id": q, "rank": int(r)}
                         for q, r in zip(image_ids, i2t.tolist()))
        rank_rows.extend({"code": code, "seed": seed, "direction": "t2i", "query_id": q, "rank": int(r)}
                         for q, r in zip(caption_ids, t2i.tolist()))

    # ---------------------------------------------------------- E1, E2, E3
    for code, loss_name in (("E1", "infonce"), ("E2", "triplet")):
        for seed in config.SEEDS:
            model = load_dual_encoder(data, loss_name, seed, device)
            img_emb, txt_emb = embed_split(model, data, "test")
            record(code, seed, *ranks_from_scores(img_emb @ txt_emb.T, owner))

            if seed == config.SEEDS[0]:   # latency: encode one query through its tower, score gallery, top-10
                feats_txt, feats_img = data["text"][test["cap"]], data["image"][test["img"]]
                latency[(code, "t2i")] = query_latency_ms(
                    device, lambda q: (model.embed_text(feats_txt[q:q + 1]) @ img_emb.T).topk(k), len(caption_ids))
                latency[(code, "i2t")] = query_latency_ms(
                    device, lambda q: (model.embed_image(feats_img[q:q + 1]) @ txt_emb.T).topk(k), len(image_ids))

            if code == "E1":
                reranker = MatchMLP().to(device)
                reranker.load_state_dict(torch.load(CHECKPOINTS / f"rerank_seed{seed}.pt", map_location=device))
                reranker.eval()
                i2t, t2i, n_calls = rerank_ranks(reranker, img_emb, txt_emb, owner)
                record("E3", seed, i2t, t2i)
                if seed == config.SEEDS[0]:
                    def rr_t2i(q):
                        t = model.embed_text(feats_txt[q:q + 1])
                        top = (t @ img_emb.T).topk(k).indices[0]
                        return reranker(img_emb[top], t.expand(k, -1)).argsort(descending=True)

                    def rr_i2t(q):
                        im = model.embed_image(feats_img[q:q + 1])
                        top = (im @ txt_emb.T).topk(k).indices[0]
                        return reranker(im.expand(k, -1), txt_emb[top]).argsort(descending=True)
                    latency[("E3", "t2i")] = query_latency_ms(device, rr_t2i, len(caption_ids))
                    latency[("E3", "i2t")] = query_latency_ms(device, rr_i2t, len(image_ids))
        print(f"  {NAMES[code]} done", flush=True)

    # ---------------------------------------------------------- E4 CLIP (embeddings from step 05)
    clip = torch.load(CLIP_FILE, weights_only=False)["test"]
    assert clip["image_ids"] == image_ids and clip["caption_ids"] == caption_ids, "CLIP test order mismatch"
    c_img, c_txt = clip["image_emb"].to(device), clip["text_emb"].to(device)
    record("E4", 0, *ranks_from_scores(c_img @ c_txt.T, clip["owner"]))
    # CLIP latency: scoring only, from precomputed embeddings (its encoders are not timed here).
    latency[("E4", "t2i")] = query_latency_ms(device, lambda q: (c_txt[q:q + 1] @ c_img.T).topk(k), len(caption_ids))
    latency[("E4", "i2t")] = query_latency_ms(device, lambda q: (c_img[q:q + 1] @ c_txt.T).topk(k), len(image_ids))
    print("  E4 CLIP done", flush=True)

    # ---------------------------------------------------------- tables
    df = pd.DataFrame(per_seed)
    save_csv(df, config.RESULTS / "06_test_per_seed.csv")
    save_csv(pd.DataFrame(rank_rows), config.RESULTS / "06_test_ranks.csv")

    metric_cols = [c for c in df.columns if c not in ("code", "method", "seed")]
    summary = df.groupby(["code", "method"])[metric_cols].agg(["mean", "std"]).round(3)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    summary.insert(2, "n_seeds", df.groupby("code").size().values)
    summary["meets_proposal_targets"] = ((summary["i2t_R@1_mean"] >= TARGET_R1) & (summary["t2i_R@1_mean"] >= TARGET_R1)
                                         & (summary["i2t_R@5_mean"] >= TARGET_R5) & (summary["t2i_R@5_mean"] >= TARGET_R5))
    save_csv(summary, config.RESULTS / "06_test_summary.csv")

    rb = random_baseline(len(image_ids), captions_per_image=5)
    save_json({"n_test_images": len(image_ids), "n_test_captions": len(caption_ids),
               "note": "expected recall (%) for random ranking; i2t assumes 5 captions per image",
               "recall_percent": rb,
               "proposal_targets_percent": {"R@1": TARGET_R1, "R@5": TARGET_R5}},
              config.RESULTS / "06_random_baseline.json")

    runs03 = pd.read_csv(config.RESULTS / "03_runs.csv")
    over = []
    for code, loss_name in (("E1", "infonce"), ("E2", "triplet")):
        r = runs03[runs03["loss"] == loss_name]
        t = df[df["code"] == code]
        over.append({"code": code, "method": NAMES[code],
                     "train_probe_rsum_mean": round(r["train_probe_rsum"].mean(), 2),
                     "val_rsum_mean": round(r["val_rsum"].mean(), 2),
                     "test_rsum_mean": round(t["rsum"].mean(), 2),
                     "train_minus_test_rsum": round(r["train_probe_rsum"].mean() - t["rsum"].mean(), 2),
                     "val_minus_test_rsum": round(r["val_rsum"].mean() - t["rsum"].mean(), 2)})
    save_csv(pd.DataFrame(over), config.RESULTS / "06_overfitting.csv")

    runs04 = pd.read_csv(config.RESULTS / "04_rerank_runs.csv")
    runs04 = runs04[runs04["method"].str.startswith("E3")]
    clip_json = load_json(config.RESULTS / "05_clip.json")
    dual_params = int(runs03["n_parameters"].iloc[0])
    eff = [
        {"code": "E1", "trainable_parameters": dual_params,
         "train_seconds_mean": round(runs03[runs03["loss"] == "infonce"]["train_seconds"].mean(), 2),
         "train_peak_vram_gb_max": runs03[runs03["loss"] == "infonce"]["peak_vram_gb"].max()},
        {"code": "E2", "trainable_parameters": dual_params,
         "train_seconds_mean": round(runs03[runs03["loss"] == "triplet"]["train_seconds"].mean(), 2),
         "train_peak_vram_gb_max": runs03[runs03["loss"] == "triplet"]["peak_vram_gb"].max()},
        {"code": "E3", "trainable_parameters": dual_params + int(runs04["n_parameters"].iloc[0]),
         "train_seconds_mean": round(runs03[runs03["loss"] == "infonce"]["train_seconds"].mean()
                                     + runs04["train_seconds"].mean(), 2),
         "train_peak_vram_gb_max": max(runs03["peak_vram_gb"].max(), runs04["peak_vram_gb"].max())},
        {"code": "E4", "trainable_parameters": 0, "train_seconds_mean": 0.0,
         "train_peak_vram_gb_max": None},
    ]
    for row in eff:
        row["method"] = NAMES[row["code"]]
        row["ms_per_query_t2i"] = latency[(row["code"], "t2i")]
        row["ms_per_query_i2t"] = latency[(row["code"], "i2t")]
    eff_df = pd.DataFrame(eff)
    eff_df["note"] = ["frozen ResNet-50 + BERT features precomputed (step 02); query passes through a tower"] * 2 + \
        [f"as E1, plus {k} MLP scores per query"] + \
        [f"scoring only from precomputed CLIP embeddings; CLIP encoding peak VRAM {clip_json['test_peak_vram_gb']} GB (step 05)"]
    save_csv(eff_df, config.RESULTS / "06_efficiency.csv")

    run_plots("06")   # separate process, see src/plots.py

    print("\nStep 06 done. TEST results (mean over seeds; E4 single run):")
    for _, r in summary.iterrows():
        print(f"  {r['code']}: rsum {r['rsum_mean']} | R@1 i2t {r['i2t_R@1_mean']} t2i {r['t2i_R@1_mean']} | "
              f"R@5 i2t {r['i2t_R@5_mean']} t2i {r['t2i_R@5_mean']} | targets met: {r['meets_proposal_targets']}")
    print(f"  random: R@1 {rb['t2i_R@1']}%, R@5 {rb['t2i_R@5']}%")


if __name__ == "__main__":
    main()
