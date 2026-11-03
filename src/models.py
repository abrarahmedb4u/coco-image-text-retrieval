"""Models trained on the cached features.

DualEncoder  - two separate towers into one shared 256-d space (coordinated
               representation, W03 lecture; dual encoder, W08 practical).
MatchMLP     - concatenation + MLP that scores one (image, caption) pair
               (early/joint fusion, W03/W04 lectures; two-tower match
               classifier, W06-07 practical). Used as a re-ranker (E3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def _tower(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class DualEncoder(nn.Module):
    def __init__(self, image_dim=2048, text_dim=768, hidden=config.HIDDEN_DIM, embed=config.EMBED_DIM):
        super().__init__()
        self.image_tower = _tower(image_dim, hidden, embed)
        self.text_tower = _tower(text_dim, hidden, embed)

    def embed_image(self, x):
        return F.normalize(self.image_tower(x), dim=-1)   # unit length -> dot product = cosine

    def embed_text(self, x):
        return F.normalize(self.text_tower(x), dim=-1)

    def forward(self, images, texts):
        return self.embed_image(images), self.embed_text(texts)


class MatchMLP(nn.Module):
    """Concatenate [image, caption, image*caption] and predict 'is this a matching pair?'."""
    def __init__(self, embed=config.EMBED_DIM, hidden=config.RERANK_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * embed, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, image_emb, text_emb):
        return self.net(torch.cat([image_emb, text_emb, image_emb * text_emb], dim=-1)).squeeze(-1)
