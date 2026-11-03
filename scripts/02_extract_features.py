"""Step 02 - extract frozen ResNet-50 image features and mean-pooled BERT caption features.

Runs each pretrained encoder once over all 5,000 images and all kept captions
and saves the vectors. Training (step 03) then only uses these saved vectors,
which keeps GPU memory low on a 4 GB card.

Outputs
  cache/features.pt                    image (N x 2048) and caption (M x 768) features
  results/02_feature_extraction.json   sizes, time, peak GPU memory, sanity checks
Run from the repo root:  python scripts/02_extract_features.py
"""
import faulthandler
import sys
from pathlib import Path

faulthandler.enable()   # print a traceback even if a C library crashes silently
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: F401  (must come before torch - see src/__init__.py)
import torch

import config
from src.data import load_captions, load_coco_val
from src.features import FEATURES_FILE, extract_image_features, extract_text_features
from src.utils import Timer, get_device, peak_vram_gb, reset_peak_vram, save_json


def main():
    device = get_device()
    print(f"Device: {device}", flush=True)
    caps = load_captions()
    print(f"[1/5] captions loaded: {len(caps)}", flush=True)
    ds = load_coco_val(decode_images=False)
    print(f"[2/5] dataset loaded: {len(ds)} rows", flush=True)

    # One row per image, ordered by image ID.
    images = caps.drop_duplicates("image_id").sort_values("image_id")
    image_ids = images["image_id"].to_numpy()
    rows = images["row"].to_numpy()
    assert all(int(ds[int(r)]["id"]) == int(i) for r, i in zip(rows[:50], image_ids[:50])), "row/id mismatch"

    print("[3/5] row/id check passed; starting ResNet-50", flush=True)
    reset_peak_vram()
    with Timer() as t_img:
        image_feats = extract_image_features(ds, rows, device)
    vram_img = peak_vram_gb()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[4/5] image features done; starting BERT", flush=True)
    reset_peak_vram()
    with Timer() as t_txt:
        text_feats = extract_text_features(caps["caption"].tolist(), device)
    vram_txt = peak_vram_gb()

    # How many captions were cut by the token limit.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    n_tokens = [len(x) for x in tokenizer(caps["caption"].tolist())["input_ids"]]

    config.CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({
        "image_ids": torch.tensor(image_ids),
        "image_feats": image_feats,
        "caption_ids": torch.tensor(caps["caption_id"].to_numpy()),
        "caption_image_ids": torch.tensor(caps["image_id"].to_numpy()),
        "caption_split": caps["split"].tolist(),
        "captions": caps["caption"].tolist(),
        "text_feats": text_feats,
    }, FEATURES_FILE)
    print(f"[saved] {FEATURES_FILE}")

    report = {
        "device": str(device),
        "image_model": f"torchvision resnet50 ({config.IMAGE_WEIGHTS}), fc removed",
        "text_model": f"{config.TEXT_MODEL}, mean over non-padding tokens of last layer",
        "fp16_autocast": device.type == "cuda",
        "n_images": int(image_feats.shape[0]),
        "image_feature_dim": int(image_feats.shape[1]),
        "n_captions": int(text_feats.shape[0]),
        "text_feature_dim": int(text_feats.shape[1]),
        "max_tokens": config.MAX_TOKENS,
        "n_captions_truncated": int(sum(n > config.MAX_TOKENS for n in n_tokens)),
        "caption_tokens_max": int(max(n_tokens)),
        "seconds_image_extraction": round(t_img.seconds, 1),
        "seconds_text_extraction": round(t_txt.seconds, 1),
        "images_per_second": round(len(rows) / t_img.seconds, 1),
        "captions_per_second": round(len(caps) / t_txt.seconds, 1),
        "peak_vram_gb_resnet50": vram_img,
        "peak_vram_gb_bert": vram_txt,
        "any_nan_image": bool(torch.isnan(image_feats).any()),
        "any_nan_text": bool(torch.isnan(text_feats).any()),
        "image_feature_norm_mean": round(image_feats.norm(dim=1).mean().item(), 3),
        "text_feature_norm_mean": round(text_feats.norm(dim=1).mean().item(), 3),
    }
    save_json(report, config.RESULTS / "02_feature_extraction.json")

    print("\nStep 02 done.")
    print(f"  images {report['n_images']} x {report['image_feature_dim']}, "
          f"captions {report['n_captions']} x {report['text_feature_dim']}")
    print(f"  time: images {report['seconds_image_extraction']}s, captions {report['seconds_text_extraction']}s")
    print(f"  peak VRAM: ResNet-50 {vram_img} GB, BERT {vram_txt} GB | NaN: "
          f"{report['any_nan_image'] or report['any_nan_text']}")


if __name__ == "__main__":
    main()
