"""Preparing the cached features and training the dual encoder (E1 / E2)."""
import copy

import torch

import config
from src.data import load_split
from src.features import load_features
from src.losses import LOSSES
from src.metrics import retrieval_metrics
from src.models import DualEncoder
from src.utils import Timer, peak_vram_gb, reset_peak_vram, set_seed

TRAIN_PROBE_SIZE = 500   # fixed training images scored like the validation set (W08 practical)


def prepare_data(device) -> dict:
    """Load features, standardise with TRAINING statistics only (W08 practical), index splits.

    Returns tensors on `device` and, for each split, the image positions,
    caption positions and each caption's owner (position within the split).
    """
    f = load_features()
    split_ids = load_split()
    image_ids = f["image_ids"].tolist()
    pos_of = {iid: k for k, iid in enumerate(image_ids)}
    cap_img_pos = torch.tensor([pos_of[int(i)] for i in f["caption_image_ids"].tolist()])

    def standardise(x, rows):
        mean, std = x[rows].mean(0), x[rows].std(0).clamp_min(1e-6)
        return (x - mean) / std

    train_img = torch.tensor(sorted(pos_of[i] for i in split_ids["train"]))
    train_cap = torch.nonzero(torch.isin(cap_img_pos, train_img)).flatten()
    image = standardise(f["image_feats"], train_img).to(device)
    text = standardise(f["text_feats"], train_cap).to(device)

    g = torch.Generator().manual_seed(123)
    probe = train_img[torch.randperm(len(train_img), generator=g)[:TRAIN_PROBE_SIZE]].sort().values

    splits = {}
    for name, img_pos in (("train", train_img),
                          ("train_probe", probe),
                          ("val", torch.tensor(sorted(pos_of[i] for i in split_ids["val"]))),
                          ("test", torch.tensor(sorted(pos_of[i] for i in split_ids["test"])))):
        cap_pos = torch.nonzero(torch.isin(cap_img_pos, img_pos)).flatten()
        local = {int(p): k for k, p in enumerate(img_pos.tolist())}
        owner = torch.tensor([local[int(p)] for p in cap_img_pos[cap_pos].tolist()])
        first_cap = torch.tensor([cap_pos[owner == k][0] for k in range(len(img_pos))])
        splits[name] = {"img": img_pos, "cap": cap_pos, "owner": owner, "first_cap": first_cap}

    return {"image": image, "text": text, "cap_img_pos": cap_img_pos, "image_ids": image_ids,
            "captions": f["captions"], "splits": splits}


def captions_by_image(data, split="train"):
    """Padded (n_images, max_caps) table of caption positions + counts, for random sampling."""
    s = data["splits"][split]
    counts = torch.bincount(s["owner"], minlength=len(s["img"]))
    table = torch.full((len(s["img"]), int(counts.max())), -1, dtype=torch.long)
    fill = torch.zeros(len(s["img"]), dtype=torch.long)
    for cap, own in zip(s["cap"].tolist(), s["owner"].tolist()):
        table[own, fill[own]] = cap
        fill[own] += 1
    return table, counts


@torch.no_grad()
def embed_split(model, data, split):
    s = data["splits"][split]
    model.eval()
    return model.embed_image(data["image"][s["img"]]), model.embed_text(data["text"][s["cap"]])


@torch.no_grad()
def evaluate_split(model, data, split, loss_fn):
    """Retrieval metrics on the split's gallery + loss on one fixed caption per image."""
    s = data["splits"][split]
    img_emb, txt_emb = embed_split(model, data, split)
    metrics = retrieval_metrics(img_emb @ txt_emb.T, s["owner"])
    probe_loss = loss_fn(img_emb, model.embed_text(data["text"][s["first_cap"]])).item()
    return metrics, probe_loss


def train_dual_encoder(data, loss_name, seed, device):
    """Train one model; keep the epoch with the best validation R-sum; stop after PATIENCE bad epochs."""
    set_seed(seed)
    loss_fn = LOSSES[loss_name]
    model = DualEncoder(data["image"].shape[1], data["text"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    table, counts = captions_by_image(data, "train")
    train_img = data["splits"]["train"]["img"]
    gen = torch.Generator().manual_seed(seed)

    history, best, best_state, bad_epochs = [], None, None, 0
    reset_peak_vram()

    def log_epoch(epoch, train_loss, seconds):
        val_m, val_loss = evaluate_split(model, data, "val", loss_fn)
        tr_m, tr_loss = evaluate_split(model, data, "train_probe", loss_fn)
        row = {"epoch": epoch, "train_batch_loss": train_loss, "train_probe_loss": round(tr_loss, 4),
               "val_loss": round(val_loss, 4), "train_probe_rsum": tr_m["rsum"], "val_rsum": val_m["rsum"],
               "train_probe_i2t_R@1": tr_m["i2t_R@1"], "train_probe_t2i_R@1": tr_m["t2i_R@1"],
               "val_i2t_R@1": val_m["i2t_R@1"], "val_t2i_R@1": val_m["t2i_R@1"], "epoch_seconds": seconds}
        history.append(row)
        return val_m, tr_m

    log_epoch(0, None, 0.0)   # untrained model, for reference
    with Timer() as t_total:
        for epoch in range(1, config.MAX_EPOCHS + 1):
            with Timer() as t_epoch:
                model.train()
                order = torch.randperm(len(train_img), generator=gen)
                # One random caption per image per epoch: two captions of the same
                # image in one batch would otherwise count as a "wrong" pair.
                pick = (torch.rand(len(order), generator=gen) * counts[order]).long()
                caps = table[order, pick]
                losses = []
                for start in range(0, len(order), config.BATCH_SIZE):
                    b_img = train_img[order[start:start + config.BATCH_SIZE]]
                    b_cap = caps[start:start + config.BATCH_SIZE]
                    if len(b_img) < 2:
                        continue
                    optimizer.zero_grad()
                    loss = loss_fn(*model(data["image"][b_img], data["text"][b_cap]))
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
            val_m, tr_m = log_epoch(epoch, round(sum(losses) / len(losses), 4), round(t_epoch.seconds, 2))
            if best is None or val_m["rsum"] > best["val"]["rsum"]:
                best = {"epoch": epoch, "val": val_m, "train_probe": tr_m}
                best_state, bad_epochs = copy.deepcopy(model.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= config.PATIENCE:
                    break

    model.load_state_dict(best_state)
    info = {"best_epoch": best["epoch"], "epochs_run": history[-1]["epoch"],
            "train_seconds": round(t_total.seconds, 1), "peak_vram_gb": peak_vram_gb(),
            "n_parameters": sum(p.numel() for p in model.parameters())}
    return model, history, best, info
