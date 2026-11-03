"""The two training objectives compared in E1 vs E2.

Both take L2-normalised image and text embeddings of shape (batch, dim),
where row i of each comes from the same (image, caption) pair.
"""
import torch
import torch.nn.functional as F

import config


def info_nce(image_emb, text_emb, temperature=config.TEMPERATURE):
    """Symmetric InfoNCE (W03 lecture: CLIP contrastive loss; W08 practical: info_nce).

    Similarity matrix / temperature -> cross-entropy where the correct 'class'
    for row i is column i. Averaged over image->text and text->image.
    Every other item in the batch acts as a negative.
    """
    logits = image_emb @ text_emb.T / temperature
    targets = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def triplet_hardest(image_emb, text_emb, margin=config.MARGIN):
    """Max-of-hinges triplet ranking loss (Faghri et al., 2018, VSE++).

    For each matching pair, only the single HARDEST wrong item in the batch is
    used: the matching score must beat it by at least `margin`, in both directions.
    """
    scores = image_emb @ text_emb.T
    positive = scores.diag()
    eye = torch.eye(len(scores), dtype=torch.bool, device=scores.device)
    cost_text = (margin + scores - positive.unsqueeze(1)).clamp(min=0).masked_fill(eye, 0)   # image -> wrong caption
    cost_image = (margin + scores - positive.unsqueeze(0)).clamp(min=0).masked_fill(eye, 0)  # caption -> wrong image
    return (cost_text.max(dim=1).values + cost_image.max(dim=0).values).mean()


LOSSES = {"infonce": info_nce, "triplet": triplet_hardest}
