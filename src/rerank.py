"""E3: concatenation + MLP re-ranker on top of the E1 (InfoNCE) dual encoder.

Stage 1 (fast): E1 cosine similarity finds the top-K candidates.
Stage 2 (slow): MatchMLP looks at [image, caption, image*caption] for each of
those K pairs together (joint/early fusion, W03/W04) and re-orders them.
Items outside the top-K keep E1's order, so Recall@K itself cannot change.
"""
import copy

import torch
import torch.nn.functional as F

import config
from src.metrics import metrics_from_ranks, ranks_from_scores
from src.models import DualEncoder, MatchMLP
from src.training import embed_split
from src.utils import Timer, set_seed

CHECKPOINTS = config.CACHE / "checkpoints"


def load_dual_encoder(data, loss_name, seed, device):
    model = DualEncoder(data["image"].shape[1], data["text"].shape[1]).to(device)
    model.load_state_dict(torch.load(CHECKPOINTS / f"{loss_name}_seed{seed}.pt", map_location=device))
    return model.eval()


@torch.no_grad()
def mine_pairs(img_emb, txt_emb, owner, k=config.RERANK_K):
    """Positives = true (image, caption) pairs. Hard negatives = wrong items inside E1's top-K,
    in both directions, i.e. exactly the kind of mistakes the re-ranker must fix at test time."""
    owner = owner.to(img_emb.device)
    scores = img_emb @ txt_emb.T
    n_img, n_cap = scores.shape
    pos = torch.stack([owner, torch.arange(n_cap, device=owner.device)], dim=1)
    top_cap = scores.topk(k, dim=1).indices                                   # (N, k) image -> captions
    i_idx = torch.arange(n_img, device=owner.device).unsqueeze(1).expand(-1, k)
    neg_i2t = torch.stack([i_idx.reshape(-1), top_cap.reshape(-1)], dim=1)
    neg_i2t = neg_i2t[owner[neg_i2t[:, 1]] != neg_i2t[:, 0]]
    top_img = scores.topk(k, dim=0).indices.T                                 # (M, k) caption -> images
    c_idx = torch.arange(n_cap, device=owner.device).unsqueeze(1).expand(-1, k)
    neg_t2i = torch.stack([top_img.reshape(-1), c_idx.reshape(-1)], dim=1)
    neg_t2i = neg_t2i[owner[neg_t2i[:, 1]] != neg_t2i[:, 0]]
    neg = torch.unique(torch.cat([neg_i2t, neg_t2i]), dim=0)
    return pos, neg   # each row: (image index, caption index) within the split


def pair_loss(reranker, img_emb, txt_emb, pos, neg):
    """Mean binary cross-entropy over all positive + hard-negative pairs (in chunks to save GPU memory)."""
    pairs = torch.cat([pos, neg])
    labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(img_emb.device)
    total = 0.0
    for start in range(0, len(pairs), 50_000):
        p = pairs[start:start + 50_000]
        logits = reranker(img_emb[p[:, 0]], txt_emb[p[:, 1]])
        total += F.binary_cross_entropy_with_logits(logits, labels[start:start + 50_000], reduction="sum").item()
    return torch.tensor(total / len(pairs))


@torch.no_grad()
def rerank_ranks(reranker, img_emb, txt_emb, owner, k=config.RERANK_K):
    """Ranks after re-ordering each query's top-K with the MLP. Returns (i2t_rank, t2i_rank, n_mlp_calls)."""
    reranker.eval()
    scores = img_emb @ txt_emb.T
    big = scores.abs().max() + 1e4          # keeps re-ranked top-K above everything else

    rows = scores.clone()                   # image -> text: re-order each row's top-K
    top_cap = scores.topk(k, dim=1).indices
    i_idx = torch.arange(len(scores), device=scores.device).unsqueeze(1).expand(-1, k)
    logits = reranker(img_emb[i_idx.reshape(-1)], txt_emb[top_cap.reshape(-1)]).view(-1, k)
    rows.scatter_(1, top_cap, big + logits)
    i2t_rank, _ = ranks_from_scores(rows, owner)

    cols = scores.clone()                   # text -> image: re-order each column's top-K
    top_img = scores.topk(k, dim=0).indices                                   # (k, M)
    c_idx = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0).expand(k, -1)
    logits = reranker(img_emb[top_img.reshape(-1)], txt_emb[c_idx.reshape(-1)]).view(k, -1)
    cols.scatter_(0, top_img, big + logits)
    _, t2i_rank = ranks_from_scores(cols, owner)
    return i2t_rank, t2i_rank, int(top_cap.numel() + top_img.numel())


def train_reranker(data, base_model, seed, device):
    """Train MatchMLP on frozen E1 embeddings; keep the epoch with best validation R-sum after re-ranking."""
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr_img, tr_txt = embed_split(base_model, data, "train")
    va_img, va_txt = embed_split(base_model, data, "val")
    tr_owner, va_owner = data["splits"]["train"]["owner"], data["splits"]["val"]["owner"]
    tr_pos, tr_neg = mine_pairs(tr_img, tr_txt, tr_owner)
    va_pos, va_neg = mine_pairs(va_img, va_txt, va_owner)

    reranker = MatchMLP().to(device)
    opt = torch.optim.AdamW(reranker.parameters(), lr=config.RERANK_LR, weight_decay=config.WEIGHT_DECAY)
    n_neg = min(len(tr_neg), config.RERANK_NEG_PER_POS * len(tr_pos))
    history, best, best_state = [], None, None

    with Timer() as t_total:
        for epoch in range(1, config.RERANK_EPOCHS + 1):
            reranker.train()
            neg = tr_neg[torch.randperm(len(tr_neg), generator=gen)[:n_neg].to(tr_neg.device)]
            pairs = torch.cat([tr_pos, neg])
            labels = torch.cat([torch.ones(len(tr_pos)), torch.zeros(len(neg))]).to(device)
            order = torch.randperm(len(pairs), generator=gen).to(device)
            losses = []
            for start in range(0, len(order), 512):
                b = order[start:start + 512]
                opt.zero_grad()
                logit = reranker(tr_img[pairs[b, 0]], tr_txt[pairs[b, 1]])
                loss = F.binary_cross_entropy_with_logits(logit, labels[b])
                loss.backward()
                opt.step()
                losses.append(loss.item())
            reranker.eval()
            with torch.no_grad():
                val_bce = pair_loss(reranker, va_img, va_txt, va_pos, va_neg).item()
                train_bce = pair_loss(reranker, tr_img, tr_txt, tr_pos, tr_neg).item()
            val_m = metrics_from_ranks(*rerank_ranks(reranker, va_img, va_txt, va_owner)[:2])
            history.append({"epoch": epoch, "train_batch_loss": round(sum(losses) / len(losses), 4),
                            "train_bce_all_pairs": round(train_bce, 4), "val_bce_all_pairs": round(val_bce, 4),
                            "val_rsum": val_m["rsum"], "val_i2t_R@1": val_m["i2t_R@1"],
                            "val_t2i_R@1": val_m["t2i_R@1"]})
            if best is None or val_m["rsum"] > best["val"]["rsum"]:
                best = {"epoch": epoch, "val": val_m}
                best_state = copy.deepcopy(reranker.state_dict())

    reranker.load_state_dict(best_state)
    info = {"best_epoch": best["epoch"], "train_seconds": round(t_total.seconds, 1),
            "n_train_positive_pairs": len(tr_pos), "n_train_hard_negative_pool": len(tr_neg),
            "n_negatives_per_epoch": n_neg, "n_parameters": sum(p.numel() for p in reranker.parameters())}
    return reranker, history, best, info
