# Bidirectional Image–Text Retrieval on COCO Captions

Project 2.2 (individual implementation of the Group 11 proposal). Given a caption, find its image
among 500 test images; given an image, find its captions. Four approaches are compared:

| Code | Method | Fusion type | Course source |
|---|---|---|---|
| E1 | Dual encoder + symmetric **InfoNCE** | coordinated representation | W03 lecture (CLIP loss), W08 practical |
| E2 | Dual encoder + **hardest-negative triplet** loss | coordinated representation | Faghri et al. 2018 (VSE++) |
| E3 | E1 + **concatenation-MLP re-ranker** on E1's top-10 | early / joint fusion | W04 lecture, W06–07 practicals |
| E4 | **Zero-shot CLIP ViT-B/32** (reference only, not trained) | coordinated representation | W03 lecture, Radford et al. 2021 |

E1–E3 use **frozen** pretrained encoders (ResNet-50 for images, mean-pooled BERT for captions).
Their outputs are computed once and cached, so training fits easily on a 4 GB laptop GPU.

**All results:** [`results_summary.md`](results_summary.md) (every number with its source file in `results/`).

---

## 1. Setup

Tested on Windows 11, Anaconda Python 3.13, NVIDIA RTX 3050 Laptop GPU (4 GB). Exact versions:
`results/00_environment.json`.

```powershell
git clone <this repository>
cd retrieval_project
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True (CPU also works, slower)
```

Internet is needed on the first run: the COCO val split (Hugging Face `lmms-lab/COCO-Caption2017`, val files only),
ResNet-50 weights, `bert-base-uncased` and `openai/clip-vit-base-patch32` are downloaded and cached.

## 2. Reproduce everything

Run from the repository root, in order. Each step saves its metrics to `results/` (CSV/JSON) and its
figures to `results/figures/`; large intermediate files go to `cache/` (not in git).

```powershell
python scripts\01_prepare_data.py      # data-quality checks, cleaning, 4000/500/500 split by image ID
python scripts\02_extract_features.py  # frozen ResNet-50 + BERT features -> cache/features.pt
python scripts\03_train.py             # E1 InfoNCE and E2 triplet, 3 seeds each (validation only)
python scripts\04_rerank.py            # E3 concat-MLP re-ranker on each E1 model (validation only)
python scripts\05_clip_baseline.py     # E4 zero-shot CLIP embeddings (validation metrics)
python scripts\06_evaluate.py          # ALL methods on the test set (only step that uses test)
python scripts\07_analysis.py          # gender/scene/length/duplicate audits + failure cases
python scripts\08_summary.py           # writes results_summary.md from results/ only
```

Figures can be redrawn at any time without PyTorch: `python scripts\plot.py all`.

Seeds and every other setting are listed in `config.py` and saved by step 08 to `results/00_config.json`.

## 3. Repository structure

```
config.py            every path, seed and hyperparameter (with the course week each comes from)
src/
  data.py            loading, caption checks, image-level split
  features.py        frozen ResNet-50 / BERT extraction, zero-shot CLIP encoding
  models.py          DualEncoder (E1/E2), MatchMLP (E3)
  losses.py          info_nce, triplet_hardest
  metrics.py         Recall@K, MRR, median rank (both directions), random baseline
  training.py        feature standardisation, split indexing, dual-encoder training + early stopping
  rerank.py          hard-negative mining, re-ranker training, top-K re-ranking
  plots.py           all figures (no PyTorch, see "Windows notes")
  utils.py           seeding, saving, timing, GPU memory
scripts/             01-08 (run in order) + plot.py
results/             all metrics (CSV/JSON) + figures - committed
results_summary.md   generated summary of results/
cache/               features, embeddings, checkpoints - gitignored, rebuilt by the scripts
```

## 4. Method details

- **Data (step 01).** COCO Caption 2017 validation split (5,000 images). Invalid captions
  ("There is no image here…") and exact duplicate captions of the same image are removed; greyscale
  images are converted to RGB. Split 4,000 / 500 / 500 **by image ID**, so no image's captions leak
  across splits.
- **Features (step 02).** Images: resize 256, centre-crop 224, ImageNet normalisation, ResNet-50
  (torchvision `IMAGENET1K_V2`) with the classifier removed -> 2048-d. Captions: `bert-base-uncased`,
  mean of last-layer token vectors over non-padding tokens, max 40 tokens -> 768-d.
  Both standardised with training-set statistics only (W08 practical).
- **Dual encoder (E1/E2, step 03).** Two 2-layer MLP towers (-> 512 -> 256), L2-normalised;
  similarity = cosine. AdamW, batch 128, one random caption per image per epoch. The epoch with the
  best validation R-sum is kept; training stops after 5 epochs without improvement.
- **Re-ranker (E3, step 04).** E1 shortlists the top-10; an MLP scores
  `[image, caption, image*caption]` for each shortlisted pair and re-orders them. Trained with
  binary cross-entropy on true pairs vs. E1's own top-10 mistakes on the training set.
- **Evaluation (step 06).** Recall@1/5/10, MRR and median rank in both directions on the 500-image test
  gallery. Image->text counts a hit if any of the image's captions is in the top K. Ties count
  against the model. Overfitting: R-sum on 500 fixed training images vs. validation vs. test.
- **Analysis (step 07).** Text->image recall by gendered words in the query (with 95% bootstrap
  interval on the male-female R@5 gap), by scene keyword group, by caption length, and for captions
  whose text also appears on another test image; 12 worst failure cases of E1.

All hyperparameters: `config.py`.

## 5. Windows notes (problems hit and fixed)

- `datasets` raised `ExpectedMoreSplitsError` because only the val split is downloaded on purpose
  -> `verification_mode="no_checks"` in `src/data.py`.
- Importing `pyarrow.dataset` after `torch` crashed with an access violation -> every script imports
  `src` (which loads pyarrow first) before `torch` (`src/__init__.py`).
- PyTorch and Anaconda's numpy each load Intel OpenMP (`OMP: Error #15`) -> figures are drawn in a
  separate process that never imports PyTorch (`src/plots.py`, `scripts/plot.py`). The unsafe
  `KMP_DUPLICATE_LIB_OK=TRUE` workaround is **not** used.

## 6. Data, licences and ethics

COCO caption annotations are CC BY 4.0; the images keep their original Flickr licences, mostly
non-commercial and/or no-derivatives (audit: `results/01_license_counts.csv`). This repository
therefore contains **no images**, only image IDs, captions and derived metrics. Gender groups are
defined from caption words, not from people in the images, so they cannot establish demographic
fairness. See `results_summary.md`, Sections 8-10.

## 7. References

- COCO-Caption2017, lmms-lab (val split). https://huggingface.co/datasets/lmms-lab/COCO-Caption2017
- COCO Consortium. Terms of Use. https://cocodataset.org/#termsofuse
- He et al. (2016). Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
- Devlin et al. (2019). BERT. https://aclanthology.org/N19-1423/
- Radford et al. (2021). CLIP. https://proceedings.mlr.press/v139/radford21a.html
- Faghri et al. (2018). VSE++. https://arxiv.org/abs/1707.05612
- Course materials: W02, W03, W04 lectures; W06–07 practicals and lectures; W08 practical.
