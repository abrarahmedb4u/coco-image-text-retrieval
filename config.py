"""All paths, seeds and hyperparameters in one place.

Every script imports this file, so changing a value here changes it everywhere.
Comments name the course week (or reference) each choice comes from.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"            # large files: features, checkpoints (gitignored)
RESULTS = ROOT / "results"        # every metric file + figures (committed)
FIGURES = RESULTS / "figures"

# ---------------------------------------------------------------- data
# lmms-lab COCO Caption 2017, val split only (5,000 images, ~0.8 GB).
# data_files restricts the download to the two val parquet files;
# without it the 40.7k-image test split (~6.6 GB) would also be downloaded.
HF_DATASET = "lmms-lab/COCO-Caption2017"
HF_DATA_FILES = {"val": "data/val-*.parquet"}

SPLIT_SEED = 0
N_VAL = 500                       # proposal: 4,000 / 500 / 500 split by image ID
N_TEST = 500                      # train = all remaining images

# Captions containing these phrases are treated as invalid (proposal, Section 2).
INVALID_CAPTION_PATTERNS = ["there is no image here"]

# COCO licence IDs -> names, from the "licenses" field of the official COCO
# 2017 annotation file (cocodataset.org). IDs 1-7 verified against
# annotations/captions_val2017.json. Used only for the licence audit.
COCO_LICENSES = {
    1: "CC BY-NC-SA 2.0",
    2: "CC BY-NC 2.0",
    3: "CC BY-NC-ND 2.0",
    4: "CC BY 2.0",
    5: "CC BY-SA 2.0",
    6: "CC BY-ND 2.0",
    7: "No known copyright restrictions",
    8: "United States Government Work",
}

# ---------------------------------------------------------------- feature extraction
# Frozen pretrained encoders (W03 lecture: pretrained representations;
# W06-08 practicals use mean-pooled BERT features).
IMAGE_RESIZE = 256                # resize shorter side, then centre-crop
IMAGE_CROP = 224                  # proposal: 224 x 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_WEIGHTS = "IMAGENET1K_V2"   # torchvision ResNet-50 ImageNet weights (He et al., 2016 architecture)
TEXT_MODEL = "bert-base-uncased"
MAX_TOKENS = 40                   # COCO captions are ~10 words; 40 tokens is ample
EXTRACT_BATCH_IMAGES = 64
EXTRACT_BATCH_TEXT = 256

# ---------------------------------------------------------------- dual encoder (E1, E2)
# Architecture and optimiser follow the W08 practical (DualEncoder, AdamW).
HIDDEN_DIM = 512
EMBED_DIM = 256                   # proposal: 256-d shared space
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5                      # early stopping on validation R-sum
TEMPERATURE = 0.07                # InfoNCE (W08 practical)
MARGIN = 0.2                      # triplet, hardest negative (Faghri et al., 2018, VSE++)
SEEDS = [0, 1, 2]

# ---------------------------------------------------------------- re-ranker (E3)
# Concatenation + MLP match classifier (W03/W04 early fusion; W06-07 two-tower classifier).
RERANK_K = 10                     # re-order the top-10 candidates from E1
RERANK_HIDDEN = 512
RERANK_NEG_PER_POS = 3            # hard negatives per positive pair when training
RERANK_EPOCHS = 15
RERANK_LR = 1e-3

# ---------------------------------------------------------------- CLIP reference (E4)
CLIP_MODEL = "openai/clip-vit-base-patch32"   # Radford et al. (2021), zero-shot only

# ---------------------------------------------------------------- bias audit
# Proposal Section 5: Recall@5 for gendered query groups.
MALE_WORDS = {"man", "men", "boy", "boys", "male", "gentleman", "guy", "guys", "he", "his", "him"}
FEMALE_WORDS = {"woman", "women", "girl", "girls", "female", "lady", "ladies", "she", "her"}

# Scene-type error review (proposal Section 5). COCO captions have no scene labels
# in this dataset version, so coarse groups are defined by caption keywords.
SCENE_KEYWORDS = {
    "animals": {"dog", "dogs", "cat", "cats", "horse", "horses", "elephant", "elephants", "giraffe", "giraffes",
                "zebra", "zebras", "bird", "birds", "sheep", "cow", "cows", "bear", "bears"},
    "food": {"pizza", "food", "plate", "sandwich", "cake", "donut", "donuts", "banana", "bananas", "broccoli",
             "vegetables", "meal", "fruit"},
    "sports": {"tennis", "baseball", "skateboard", "skateboarder", "surfboard", "surfer", "surfing", "ski",
               "skier", "skiing", "snowboard", "frisbee", "soccer", "racket", "bat"},
    "vehicles/street": {"bus", "train", "car", "cars", "truck", "street", "motorcycle", "plane", "airplane",
                        "traffic", "road", "bike", "bicycle"},
    "indoor rooms": {"kitchen", "bathroom", "bedroom", "toilet", "sink", "bed", "couch", "room", "refrigerator"},
}
N_FAILURE_EXAMPLES = 12
BOOTSTRAP_SAMPLES = 2000
