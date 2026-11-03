"""Retrieval metrics: Recall@K and MRR in both directions (W06-07 lecture lists both).

Setting: a gallery of N images and the M captions that belong to them
(about 5 per image). `cap_owner[j]` = position (0..N-1) of caption j's image.

image -> text: a query image is a hit at K if ANY of its own captions is in the top K.
text -> image: each caption has exactly one correct image.
Rank 0 = correct item is first.
"""
import torch

KS = (1, 5, 10)


@torch.no_grad()
def ranks_from_scores(scores, cap_owner):
    """scores: (N images, M captions), higher = more similar. Returns (i2t_rank (N,), t2i_rank (M,))."""
    n_img, n_cap = scores.shape
    owner = cap_owner.to(scores.device)
    correct = owner.unsqueeze(0) == torch.arange(n_img, device=scores.device).unsqueeze(1)
    # Rank = number of WRONG items scoring >= the correct one. Ties count against
    # the model (pessimistic), e.g. identical caption texts on two different images.
    best_correct = scores.masked_fill(~correct, float("-inf")).max(dim=1).values
    i2t_rank = ((scores >= best_correct.unsqueeze(1)) & ~correct).sum(dim=1)
    correct_score = scores[owner, torch.arange(n_cap, device=scores.device)]
    t2i_rank = ((scores >= correct_score.unsqueeze(0)) & ~correct).sum(dim=0)
    return i2t_rank.cpu(), t2i_rank.cpu()


def metrics_from_ranks(i2t_rank, t2i_rank) -> dict:
    """Recall values as percentages (0-100); MRR as 0-1."""
    out = {}
    for name, ranks in (("i2t", i2t_rank), ("t2i", t2i_rank)):
        for k in KS:
            out[f"{name}_R@{k}"] = round(100 * (ranks < k).float().mean().item(), 2)
        out[f"{name}_MRR"] = round((1.0 / (ranks.float() + 1)).mean().item(), 4)
        out[f"{name}_median_rank"] = float(ranks.float().median().item() + 1)   # 1-based
    out["rsum"] = round(sum(out[f"{d}_R@{k}"] for d in ("i2t", "t2i") for k in KS), 2)
    return out


def retrieval_metrics(scores, cap_owner) -> dict:
    return metrics_from_ranks(*ranks_from_scores(scores, cap_owner))


def random_baseline(n_images, captions_per_image=5) -> dict:
    """Expected Recall@K for random ranking (no model). Percentages."""
    out = {}
    for k in KS:
        out[f"t2i_R@{k}"] = round(100 * k / n_images, 2)
        # i2t: hit if any of c correct captions lands in the top k of M = c*N captions.
        m, c = n_images * captions_per_image, captions_per_image
        p_miss = 1.0
        for i in range(k):
            p_miss *= (m - c - i) / (m - i)
        out[f"i2t_R@{k}"] = round(100 * (1 - p_miss), 2)
    return out
