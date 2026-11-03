"""Step 07 - analysis on the TEST set: bias check, scene groups, caption length,
duplicate captions, and failure cases (W06-07 ethics & bias lecture; proposal Section 5).

Uses the per-query ranks saved by step 06 (mean over the 3 seeds for E1-E3).
Text -> image queries are captions, so caption words define the groups.

Outputs
  results/07_gender_audit.csv        recall per gendered caption group, per method
  results/07_gender_gap.csv          male - female R@5 with 95% bootstrap interval
  results/07_scene_groups.csv        recall per keyword scene group + how often it appears in training
  results/07_caption_length.csv      recall by caption length
  results/07_duplicate_captions.csv  captions whose text also appears on another test image vs the rest
  results/07_failure_cases.csv       E1 (seed 0) text->image misses with the captions of the wrongly retrieved image
  results/figures/07_bias_and_groups.png
Run from the repo root:  python scripts/07_analysis.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: F401  (must come before torch - see src/__init__.py)
import numpy as np
import pandas as pd
import torch

import config
from src.data import load_captions, normalise_for_duplicates
from src.features import CLIP_FILE
from src.rerank import load_dual_encoder
from src.training import embed_split, prepare_data
from src.utils import get_device, run_plots, save_csv

CODES = ["E1", "E2", "E3", "E4"]


def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def recall_table(hits, group_col, groups=None):
    """hits: one row per (code, caption query) with hit@k already averaged over seeds."""
    rows = []
    for code in CODES:
        h = hits[hits["code"] == code]
        for g, part in h.groupby(group_col, observed=False):
            if groups is not None and g not in groups:
                continue
            rows.append({"code": code, group_col: g, "n_queries": len(part),
                         "R@1": round(100 * part["hit1"].mean(), 2), "R@5": round(100 * part["hit5"].mean(), 2),
                         "R@10": round(100 * part["hit10"].mean(), 2)})
    return pd.DataFrame(rows)


def main():
    caps = load_captions()
    ranks = pd.read_csv(config.RESULTS / "06_test_ranks.csv")
    t2i = ranks[ranks["direction"] == "t2i"].copy()
    for k in (1, 5, 10):
        t2i[f"hit{k}"] = (t2i["rank"] < k).astype(float)
    # Average each query's hits over seeds -> one row per (method, caption).
    hits = t2i.groupby(["code", "query_id"])[["hit1", "hit5", "hit10"]].mean().reset_index()
    text = caps.set_index("caption_id")["caption"]
    hits["caption"] = hits["query_id"].map(text)
    hits["words"] = hits["caption"].map(words)

    # ---------------------------------------------------------- gender audit
    def gender(ws):
        m, f = bool(ws & config.MALE_WORDS), bool(ws & config.FEMALE_WORDS)
        return "both" if m and f else "male terms" if m else "female terms" if f else "neither"
    hits["gender_group"] = hits["words"].map(gender)
    save_csv(recall_table(hits, "gender_group"), config.RESULTS / "07_gender_audit.csv")

    rng = np.random.default_rng(0)
    gap_rows = []
    for code in CODES:
        h = hits[hits["code"] == code]
        male = h[h["gender_group"] == "male terms"]["hit5"].to_numpy()
        female = h[h["gender_group"] == "female terms"]["hit5"].to_numpy()
        boots = [rng.choice(male, len(male)).mean() - rng.choice(female, len(female)).mean()
                 for _ in range(config.BOOTSTRAP_SAMPLES)]
        gap_rows.append({"code": code, "n_male": len(male), "n_female": len(female),
                         "R@5_male": round(100 * male.mean(), 2), "R@5_female": round(100 * female.mean(), 2),
                         "gap_male_minus_female": round(100 * (male.mean() - female.mean()), 2),
                         "ci95_low": round(100 * np.percentile(boots, 2.5), 2),
                         "ci95_high": round(100 * np.percentile(boots, 97.5), 2)})
    gap = pd.DataFrame(gap_rows)
    gap["ci_excludes_zero"] = (gap["ci95_low"] > 0) | (gap["ci95_high"] < 0)
    save_csv(gap, config.RESULTS / "07_gender_gap.csv")

    # ---------------------------------------------------------- scene groups (a caption can be in several)
    train_words = caps[caps["split"] == "train"]["caption"].map(words)
    scene_rows = []
    for scene, kw in config.SCENE_KEYWORDS.items():
        in_group = hits["words"].map(lambda ws: bool(ws & kw))
        n_train = int(train_words.map(lambda ws: bool(ws & kw)).sum())
        for code in CODES:
            part = hits[(hits["code"] == code) & in_group]
            scene_rows.append({"scene_group": scene, "code": code, "n_test_queries": len(part),
                               "n_train_captions_with_keyword": n_train,
                               "share_of_train_captions_pct": round(100 * n_train / len(train_words), 2),
                               "R@1": round(100 * part["hit1"].mean(), 2), "R@5": round(100 * part["hit5"].mean(), 2)})
    save_csv(pd.DataFrame(scene_rows), config.RESULTS / "07_scene_groups.csv")

    # ---------------------------------------------------------- caption length
    n_words = hits["caption"].str.split().str.len()
    hits["length_group"] = pd.cut(n_words, bins=[0, 9, 11, 1000], labels=["<=9 words", "10-11 words", ">=12 words"])
    save_csv(recall_table(hits, "length_group"), config.RESULTS / "07_caption_length.csv")

    # ---------------------------------------------------------- duplicate caption texts inside the test gallery
    test_caps = caps[caps["split"] == "test"].copy()
    test_caps["norm"] = test_caps["caption"].map(normalise_for_duplicates)
    imgs_per_text = test_caps.groupby("norm")["image_id"].nunique()
    shared = set(test_caps.loc[test_caps["norm"].map(imgs_per_text) > 1, "caption_id"])
    hits["duplicate_text"] = hits["query_id"].isin(shared).map({True: "text shared with another test image",
                                                                 False: "unique text"})
    save_csv(recall_table(hits, "duplicate_text"), config.RESULTS / "07_duplicate_captions.csv")

    # ---------------------------------------------------------- failure cases (E1 seed 0, text -> image)
    device = get_device()
    data = prepare_data(device)
    test = data["splits"]["test"]
    model = load_dual_encoder(data, "infonce", config.SEEDS[0], device)
    img_emb, txt_emb = embed_split(model, data, "test")
    scores = (txt_emb @ img_emb.T).cpu()                        # (captions, images)
    owner = test["owner"]
    image_ids = [data["image_ids"][int(p)] for p in test["img"]]
    caption_ids = test["cap"].tolist()
    correct = scores[torch.arange(len(owner)), owner]
    top1 = scores.argmax(dim=1)
    margin = scores.max(dim=1).values - correct                  # how far the wrong winner beat the right image
    clip = torch.load(CLIP_FILE, weights_only=False)["test"]
    clip_scores = clip["text_emb"] @ clip["image_emb"].T
    clip_rank = ((clip_scores >= clip_scores[torch.arange(len(owner)), clip["owner"]].unsqueeze(1))
                 & (torch.arange(len(image_ids)).unsqueeze(0) != clip["owner"].unsqueeze(1))).sum(dim=1)
    e1_rank = ((scores >= correct.unsqueeze(1)) & (torch.arange(len(image_ids)).unsqueeze(0)
                                                   != owner.unsqueeze(1))).sum(dim=1)
    by_image = caps[caps["split"] == "test"].groupby("image_id")["caption"].apply(list)
    worst = torch.argsort(margin, descending=True)[:config.N_FAILURE_EXAMPLES].tolist()
    fails = []
    for j in worst:
        right, wrong = image_ids[int(owner[j])], image_ids[int(top1[j])]
        fails.append({"caption_id": caption_ids[j], "query_caption": text[caption_ids[j]],
                      "correct_image_id": right, "E1_rank_of_correct (0=first)": int(e1_rank[j]),
                      "CLIP_rank_of_correct (0=first)": int(clip_rank[j]),
                      "retrieved_image_id": wrong, "score_margin_wrong_minus_right": round(float(margin[j]), 4),
                      "retrieved_image_caption_1": by_image[wrong][0], "retrieved_image_caption_2": by_image[wrong][1],
                      "correct_image_other_caption": next(c for c in by_image[right] if c != text[caption_ids[j]])})
    save_csv(pd.DataFrame(fails), config.RESULTS / "07_failure_cases.csv")

    run_plots("07")   # separate process, see src/plots.py

    g = gap.set_index("code")
    print("\nStep 07 done.")
    for code in CODES:
        print(f"  {code}: R@5 male-terms {g.loc[code, 'R@5_male']} vs female-terms {g.loc[code, 'R@5_female']} "
              f"| gap {g.loc[code, 'gap_male_minus_female']} [{g.loc[code, 'ci95_low']}, {g.loc[code, 'ci95_high']}]")
    print(f"  failure cases saved: {len(fails)} (open results/07_failure_cases.csv)")


if __name__ == "__main__":
    main()
