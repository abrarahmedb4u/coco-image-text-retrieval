"""Loading the COCO captions, checking caption quality, and the image-level split."""
import io
import re

import numpy as np
import pandas as pd

import config


def load_coco_val(decode_images: bool = False):
    """Load the 5,000-image COCO 2017 val set from Hugging Face.

    Images are left as raw bytes (decode_images=False) so that a broken file
    can be caught and counted instead of crashing the script.
    """
    from datasets import Image, load_dataset
    # verification_mode="no_checks": the dataset card lists a "test" split too;
    # we only download "val" on purpose, so the split-count check must be skipped.
    ds = load_dataset(config.HF_DATASET, data_files=config.HF_DATA_FILES, split="val",
                      verification_mode="no_checks")
    if not decode_images:
        ds = ds.cast_column("image", Image(decode=False))
    return ds


def open_image(image_field):
    """Raw dataset image field -> RGB PIL image (greyscale/CMYK converted to RGB)."""
    from PIL import Image as PILImage
    data = image_field["bytes"] if isinstance(image_field, dict) else image_field
    img = PILImage.open(io.BytesIO(data))
    img.load()
    return img


def clean_caption(text) -> str:
    """Strip and collapse whitespace. Case is kept (BERT-uncased lowercases anyway)."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalise_for_duplicates(text: str) -> str:
    """Lower-case, drop trailing punctuation: used only to detect duplicate captions."""
    return re.sub(r"[\s\.\!]+$", "", clean_caption(text).lower())


def invalid_reason(caption: str):
    """Return why a caption is unusable, or None if it is fine."""
    if caption == "":
        return "blank"
    lower = caption.lower()
    for pattern in config.INVALID_CAPTION_PATTERNS:
        if pattern in lower:
            return f"matches '{pattern}'"
    return None


def make_split(image_ids, seed=config.SPLIT_SEED):
    """Shuffle image IDs with a fixed seed: first N_TEST = test, next N_VAL = val, rest = train.

    Splitting by image (not by caption) means no image's captions appear in two splits.
    """
    ids = np.array(sorted(image_ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    ids = ids[order]
    test = ids[:config.N_TEST]
    val = ids[config.N_TEST:config.N_TEST + config.N_VAL]
    train = ids[config.N_TEST + config.N_VAL:]
    return {"train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist())}


# ---------------------------------------------------------------- reading what 01 saved
CAPTIONS_FILE = config.CACHE / "captions.csv"
SPLIT_FILE = config.RESULTS / "01_split_ids.json"


def load_captions() -> pd.DataFrame:
    """Columns: image_id, row (index in the HF dataset), caption_id, caption, split."""
    return pd.read_csv(CAPTIONS_FILE, encoding="utf-8")


def load_split() -> dict:
    from src.utils import load_json
    return load_json(SPLIT_FILE)
