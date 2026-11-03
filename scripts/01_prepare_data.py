"""Step 01 - data quality assessment and image-level split (W02: modality quality issues).

Reads the 5,000 COCO val images + captions, checks them, removes unusable
captions, and fixes a 4,000 / 500 / 500 split by image ID.

Outputs
  results/00_environment.json      Python / PyTorch / GPU versions
  results/01_data_quality.json     all data-quality numbers
  results/01_split_summary.csv     images, captions, caption length per split
  results/01_license_counts.csv    licence audit per split
  results/01_removed_captions.csv  every caption removed and why
  results/01_split_ids.json        image IDs in each split (no images)
  results/figures/01_data_overview.png
  cache/captions.csv               cleaned captions used by later steps
Run from the repo root:  python scripts/01_prepare_data.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: F401  (must come before torch - see src/__init__.py)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.data import (CAPTIONS_FILE, SPLIT_FILE, clean_caption, invalid_reason,
                      load_coco_val, make_split, normalise_for_duplicates, open_image)
from src.utils import Timer, device_info, save_csv, save_figure, save_json


def library_versions() -> dict:
    import datasets, matplotlib, pandas, PIL, torch, torchvision, transformers
    return {m.__name__: m.__version__ for m in (datasets, matplotlib, pandas, PIL, torch, torchvision, transformers)}


def describe(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": round(v.mean(), 2), "median": float(np.median(v)), "std": round(v.std(), 2),
            "min": float(v.min()), "max": float(v.max())}


def main():
    save_json({**device_info(), "libraries": library_versions()}, config.RESULTS / "00_environment.json")

    with Timer() as t_load:
        ds = load_coco_val(decode_images=False)
    print(f"Loaded {len(ds)} rows in {t_load.seconds:.1f}s")

    # ---------------------------------------------------------- images
    image_rows, unreadable = [], []
    with Timer() as t_scan:
        for row_idx, item in enumerate(ds):
            record = {"row": row_idx, "image_id": int(item["id"]), "license": int(item["license"]),
                      "meta_width": int(item["width"]), "meta_height": int(item["height"])}
            try:
                img = open_image(item["image"])
                record.update(mode=img.mode, width=img.size[0], height=img.size[1])
            except Exception as err:  # corrupt or unreadable file
                unreadable.append({"image_id": record["image_id"], "error": repr(err)})
                record.update(mode=None, width=None, height=None)
            record["answers"] = item["answer"]
            image_rows.append(record)
    images = pd.DataFrame(image_rows)
    readable = images[images["mode"].notna()]
    size_mismatch = int(((readable["width"] != readable["meta_width"]) |
                         (readable["height"] != readable["meta_height"])).sum())

    # ---------------------------------------------------------- captions
    caption_rows, removed = [], []
    for rec in image_rows:
        seen = set()
        for raw in rec["answers"]:
            cap = clean_caption(raw)
            reason = invalid_reason(cap)
            norm = normalise_for_duplicates(cap)
            if reason is None and norm in seen:
                reason = "duplicate within same image"
            if reason is not None:
                removed.append({"image_id": rec["image_id"], "caption": cap, "reason": reason})
                continue
            seen.add(norm)
            caption_rows.append({"image_id": rec["image_id"], "row": rec["row"], "caption": cap, "norm": norm})
    caps = pd.DataFrame(caption_rows)
    raw_counts = [len(r["answers"]) for r in image_rows]
    raw_lengths = [len(clean_caption(c).split()) for r in image_rows for c in r["answers"]]

    # Same caption text on two or more different images -> ambiguous for text->image retrieval.
    images_per_text = caps.groupby("norm")["image_id"].nunique()
    shared_texts = images_per_text[images_per_text > 1]

    # Images with no valid caption left, or unreadable, are dropped before splitting.
    unusable = set(r["image_id"] for r in unreadable) | (set(images["image_id"]) - set(caps["image_id"]))
    usable_ids = sorted(set(images["image_id"]) - unusable)
    split = make_split(usable_ids)
    split_of = {i: s for s, ids in split.items() for i in ids}

    caps = caps[caps["image_id"].isin(split_of)].copy()
    caps["split"] = caps["image_id"].map(split_of)
    caps["caption_id"] = np.arange(len(caps))
    config.CACHE.mkdir(parents=True, exist_ok=True)
    caps[["image_id", "row", "caption_id", "caption", "split"]].to_csv(CAPTIONS_FILE, index=False, encoding="utf-8")
    print(f"[saved] {CAPTIONS_FILE}")
    save_json(split, SPLIT_FILE)

    # Leakage check: no image ID in two splits.
    overlap = (len(set(split["train"]) & set(split["val"])) + len(set(split["train"]) & set(split["test"]))
               + len(set(split["val"]) & set(split["test"])))

    # Duplicate texts inside the test gallery (affects text->image evaluation).
    test_caps = caps[caps["split"] == "test"]
    test_ipt = test_caps.groupby("norm")["image_id"].nunique()
    test_ambiguous = int(test_caps["norm"].isin(test_ipt[test_ipt > 1].index).sum())

    quality = {
        "n_images": len(images),
        "n_unreadable_images": len(unreadable),
        "unreadable_images": unreadable,
        "n_images_size_differs_from_metadata": size_mismatch,
        "image_modes": dict(Counter(readable["mode"])),
        "image_width": describe(readable["width"]),
        "image_height": describe(readable["height"]),
        "n_captions_raw": int(sum(raw_counts)),
        "captions_per_image_raw": {str(k): v for k, v in sorted(Counter(raw_counts).items())},
        "caption_words_raw": describe(raw_lengths),
        "n_captions_removed": len(removed),
        "removed_by_reason": dict(Counter(r["reason"] for r in removed)),
        "n_captions_kept": int(len(caps)),
        "caption_words_kept": describe(caps["caption"].str.split().str.len()),
        "n_caption_texts_shared_by_2plus_images": int(len(shared_texts)),
        "n_captions_whose_text_appears_on_another_image": int(caps["norm"].isin(shared_texts.index).sum()),
        "n_test_captions_whose_text_appears_on_another_test_image": test_ambiguous,
        "n_images_dropped_before_split": len(unusable),
        "split_seed": config.SPLIT_SEED,
        "split_image_overlap_count": overlap,
        "seconds_load": round(t_load.seconds, 1),
        "seconds_scan": round(t_scan.seconds, 1),
    }
    save_json(quality, config.RESULTS / "01_data_quality.json")
    save_csv(pd.DataFrame(removed, columns=["image_id", "caption", "reason"]),
             config.RESULTS / "01_removed_captions.csv")

    summary = []
    for name in ("train", "val", "test"):
        part = caps[caps["split"] == name]
        summary.append({"split": name, "n_images": len(split[name]), "n_captions": len(part),
                        "captions_per_image_mean": round(len(part) / len(split[name]), 3),
                        "caption_words_mean": round(part["caption"].str.split().str.len().mean(), 2)})
    save_csv(pd.DataFrame(summary), config.RESULTS / "01_split_summary.csv")

    images["split"] = images["image_id"].map(split_of).fillna("dropped")
    lic = images.groupby(["license", "split"]).size().unstack(fill_value=0).reset_index()
    lic.insert(1, "license_name", lic["license"].map(config.COCO_LICENSES).fillna("unknown"))
    lic["total"] = lic.drop(columns=["license", "license_name"]).sum(axis=1)
    save_csv(lic, config.RESULTS / "01_license_counts.csv")

    # ---------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    lengths = caps["caption"].str.split().str.len()
    axes[0].hist(lengths, bins=range(int(lengths.min()), int(lengths.max()) + 2), color="#4C72B0")
    axes[0].set(title="(a) Caption length (kept captions)", xlabel="words per caption", ylabel="number of captions")
    cpi = Counter(raw_counts)
    axes[1].bar([str(k) for k in sorted(cpi)], [cpi[k] for k in sorted(cpi)], color="#4C72B0")
    axes[1].set(title="(b) Captions per image (raw)", xlabel="captions", ylabel="number of images")
    axes[2].scatter(readable["width"], readable["height"], s=3, alpha=0.3, color="#4C72B0")
    axes[2].set(title="(c) Image size before resizing", xlabel="width (px)", ylabel="height (px)")
    fig.tight_layout()
    save_figure(fig, config.FIGURES / "01_data_overview.png")

    print("\nStep 01 done. Key checks:")
    print(f"  images: {quality['n_images']}, unreadable: {quality['n_unreadable_images']}, "
          f"captions kept: {quality['n_captions_kept']}, removed: {quality['n_captions_removed']}")
    print(f"  split sizes: train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])}, "
          f"overlap {overlap}")


if __name__ == "__main__":
    main()
