"""Step 05 - E4: zero-shot CLIP ViT-B/32 as a state-of-the-art reference (no training).

CLIP was pretrained by OpenAI on ~400M web image-text pairs with the same
contrastive loss as E1 (W03 lecture; Radford et al., 2021). Here it is used
as-is on our validation and test galleries. Only VALIDATION numbers are
reported here; test numbers for every method are produced together in step 06.

Outputs
  cache/clip_embeddings.pt   CLIP image/caption embeddings for val + test
  results/05_clip.json       validation metrics, time, peak GPU memory
Run from the repo root:  python scripts/05_clip_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: F401  (must come before torch - see src/__init__.py)
import torch

import config
from src.data import load_captions, load_coco_val, load_split
from src.features import CLIP_FILE, extract_clip
from src.metrics import retrieval_metrics
from src.utils import Timer, get_device, peak_vram_gb, reset_peak_vram, save_json


def main():
    device = get_device()
    print(f"Device: {device}", flush=True)
    caps = load_captions()
    split = load_split()
    ds = load_coco_val(decode_images=False)

    saved, report = {}, {"model": config.CLIP_MODEL, "training": "none (zero-shot)",
                         "preprocessing": "CLIP's own image processor (resize 224, centre crop, CLIP mean/std)"}
    for name in ("val", "test"):
        ids = sorted(split[name])
        part = caps[caps["image_id"].isin(ids)].sort_values("caption_id")
        rows_of = part.drop_duplicates("image_id").set_index("image_id")["row"]
        rows = [int(rows_of[i]) for i in ids]
        owner = torch.tensor(part["image_id"].map({iid: k for k, iid in enumerate(ids)}).to_numpy())

        reset_peak_vram()
        with Timer() as t:
            img_emb, txt_emb = extract_clip(ds, rows, part["caption"].tolist(), device)
        saved[name] = {"image_ids": ids, "caption_ids": part["caption_id"].tolist(), "owner": owner,
                       "image_emb": img_emb, "text_emb": txt_emb}
        report[f"{name}_n_images"] = len(ids)
        report[f"{name}_n_captions"] = len(part)
        report[f"{name}_encode_seconds"] = round(t.seconds, 1)
        report[f"{name}_peak_vram_gb"] = peak_vram_gb()
        print(f"[{name}] encoded {len(ids)} images + {len(part)} captions in {t.seconds:.1f}s", flush=True)

    torch.save(saved, CLIP_FILE)
    print(f"[saved] {CLIP_FILE}")
    val = saved["val"]
    report["val_metrics"] = retrieval_metrics(val["image_emb"] @ val["text_emb"].T, val["owner"])
    save_json(report, config.RESULTS / "05_clip.json")

    m = report["val_metrics"]
    print("\nStep 05 done. CLIP zero-shot on validation:")
    print(f"  rsum {m['rsum']} | R@1 i2t {m['i2t_R@1']} t2i {m['t2i_R@1']} | "
          f"R@5 i2t {m['i2t_R@5']} t2i {m['t2i_R@5']} | peak VRAM {report['val_peak_vram_gb']} GB")


if __name__ == "__main__":
    main()
