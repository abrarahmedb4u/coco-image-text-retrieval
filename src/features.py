"""Frozen feature extractors: ResNet-50 for images, mean-pooled BERT for captions.

Both models are only run forward once (no training), so the outputs can be
saved and every later experiment trains on the saved vectors (W08 practical
uses BERT features extracted in advance in the same way).
"""
import torch
import torch.nn as nn

import config
from src.data import open_image


def image_transform():
    """Resize shorter side to 256, centre-crop 224, ImageNet normalisation."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(config.IMAGE_RESIZE),
        transforms.CenterCrop(config.IMAGE_CROP),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def load_resnet50(device):
    """Pretrained ResNet-50 with the classification layer removed -> 2048-d vector."""
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=ResNet50_Weights[config.IMAGE_WEIGHTS])
    model.fc = nn.Identity()          # keep the 2048-d global-average-pooled features
    return model.eval().to(device)


@torch.no_grad()
def extract_image_features(ds, rows, device, batch_size=config.EXTRACT_BATCH_IMAGES):
    """rows: dataset row indices. Returns float32 tensor (len(rows), 2048)."""
    model = load_resnet50(device)
    print("  ResNet-50 loaded on", device, flush=True)
    transform = image_transform()
    out = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        images = [transform(open_image(ds[int(r)]["image"]).convert("RGB")) for r in batch_rows]
        x = torch.stack(images).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            feats = model(x)
        out.append(feats.float().cpu())
        if (start // batch_size) % 10 == 0:
            print(f"  images {start + len(batch_rows)}/{len(rows)}", flush=True)
    return torch.cat(out)


@torch.no_grad()
def extract_text_features(captions, device, batch_size=config.EXTRACT_BATCH_TEXT):
    """Mean of BERT's last-layer token vectors, ignoring padding. Returns (len(captions), 768)."""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
    model = AutoModel.from_pretrained(config.TEXT_MODEL).eval().to(device)
    out = []
    for start in range(0, len(captions), batch_size):
        batch = list(captions[start:start + batch_size])
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=config.MAX_TOKENS,
                           return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        out.append(pooled.cpu())
        if (start // batch_size) % 20 == 0:
            print(f"  captions {start + len(batch)}/{len(captions)}", flush=True)
    return torch.cat(out)


@torch.no_grad()
def extract_clip(ds, rows, captions, device, batch_size=64):
    """Zero-shot CLIP ViT-B/32 (Radford et al., 2021): its own preprocessing, no training.

    Uses the vision/text towers + projection layers directly (stable across
    transformers versions). Returns L2-normalised (image_emb, text_emb).
    """
    from transformers import AutoProcessor, CLIPModel
    processor = AutoProcessor.from_pretrained(config.CLIP_MODEL)
    model = CLIPModel.from_pretrained(config.CLIP_MODEL).eval().to(device)
    use_amp = device.type == "cuda"

    img_out = []
    for start in range(0, len(rows), batch_size):
        images = [open_image(ds[int(r)]["image"]).convert("RGB") for r in rows[start:start + batch_size]]
        pixels = processor.image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pooled = model.vision_model(pixel_values=pixels).pooler_output
            emb = model.visual_projection(pooled)
        img_out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu())

    txt_out = []
    for start in range(0, len(captions), 256):
        tokens = processor.tokenizer(list(captions[start:start + 256]), padding=True, truncation=True,
                                     max_length=77, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pooled = model.text_model(input_ids=tokens["input_ids"],
                                      attention_mask=tokens["attention_mask"]).pooler_output
            emb = model.text_projection(pooled)
        txt_out.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu())
    return torch.cat(img_out), torch.cat(txt_out)


FEATURES_FILE = config.CACHE / "features.pt"
CLIP_FILE = config.CACHE / "clip_embeddings.pt"


def load_features() -> dict:
    return torch.load(FEATURES_FILE, weights_only=False)
