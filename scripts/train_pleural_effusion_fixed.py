"""One fixed 30-epoch SHARLOCK pleural-effusion experiment on the original GPU."""

import csv
import hashlib
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from medclip import MedCLIPModel, MedCLIPProcessor, MedCLIPVisionModel
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_sharlock_100 import LABELS, load_image


TRAIN = Path("data/processed/pleural_effusion_balanced_train.csv")
VALIDATION = Path("data/subset_100/pleural_effusion_explicit_validation.csv")
ORIGINAL = Path("runs/sharlock_100_20261123_153936_125581")
REFERENCE = ORIGINAL / "pleural_effusion_validation_predictions.csv"
BACKBONE = Path("pretrained/medclip-resnet")
AFFIRMATIVE = "A pleural effusion is present."
NEGATIVE = "No pleural effusion is present."
SEED = 42
EPOCHS = 30


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_key(row):
    return row["image_id"], row["claim"]


def check_pairs(rows, split, image_count):
    labels = defaultdict(dict)
    paths = {}
    for row in rows:
        image_id, claim = pair_key(row)
        path = Path(row["image_path"])
        if row["split"] != split or path.name != image_id or not path.is_file():
            raise ValueError(f"Unexpected split or image path for {image_id}")
        if paths.setdefault(image_id, path) != path or claim in labels[image_id]:
            raise ValueError(f"Inconsistent path or duplicate claim for {image_id}")
        labels[image_id][claim] = row["verification_label"]
    if len(labels) != image_count:
        raise ValueError(f"Expected {image_count} unique {split} images")
    for pair in labels.values():
        if set(pair) != {AFFIRMATIVE, NEGATIVE} or set(pair.values()) != set(LABELS):
            raise ValueError(f"Incomplete or inconsistent claim pair in {split}")
    if Counter(pair[AFFIRMATIVE] for pair in labels.values()) != dict.fromkeys(LABELS, image_count // 2):
        raise ValueError(f"Unbalanced {split} images")
    return paths


@torch.no_grad()
def encode_claims(processor, backbone, claims):
    lengths = processor.tokenizer(claims, truncation=False)["input_ids"]
    if any(len(ids) > 128 for ids in lengths):
        raise ValueError("Claim exceeds the baseline sequence length")
    tokens = processor.tokenizer(claims, padding="max_length", max_length=128,
                                 truncation=False, return_tensors="pt")
    vectors = backbone.encode_text(tokens["input_ids"].cuda(), tokens["attention_mask"].cuda())
    return dict(zip(claims, vectors.cpu()))


@torch.no_grad()
def features_for(rows, paths, processor, backbone, text_vectors):
    image_vectors = {}
    for image_id in sorted(paths):
        inputs = processor(images=load_image(paths[image_id]), return_tensors="pt")
        image_vectors[image_id] = backbone.encode_image(inputs["pixel_values"].cuda())[0].cpu()
    visual = torch.stack([image_vectors[r["image_id"]] for r in rows])
    textual = torch.stack([text_vectors[r["claim"]] for r in rows])
    features = torch.cat([visual, textual, torch.abs(visual - textual), visual * textual], dim=1)
    if not torch.isfinite(features).all() or features.shape != (len(rows), 2048):
        raise ValueError("Invalid MedCLIP features")
    return features


def main():
    train, val, original = (read_csv(path) for path in (TRAIN, VALIDATION, REFERENCE))
    train_paths = check_pairs(train, "train", 40)
    val_paths = check_pairs(val, "validation", 4)
    if train_paths.keys() & val_paths.keys():
        raise ValueError("Training and validation share an image")
    reference = {pair_key(r): r for r in original}
    if len(original) != 8 or reference.keys() != {pair_key(r) for r in val}:
        raise ValueError("Baseline and validation keys must match one-to-one for all eight pairs")
    for r in val:
        old = reference[pair_key(r)]
        if any(old[k] != r[k] for k in ("verification_label", "source_annotation", "split")):
            raise ValueError(f"Reference metadata differs for {pair_key(r)}")

    if not torch.cuda.is_available():
        raise RuntimeError("This controlled run requires access to the original CUDA device")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    processor = MedCLIPProcessor()
    backbone = MedCLIPModel(vision_cls=MedCLIPVisionModel)
    backbone.from_pretrained(input_dir=str(BACKBONE))
    backbone.requires_grad_(False).cuda().eval()
    text_vectors = encode_claims(processor, backbone, sorted({r["claim"] for r in train}))
    x = features_for(train, train_paths, processor, backbone, text_vectors)
    y = torch.tensor([LABELS[r["verification_label"]] for r in train], dtype=torch.long)

    head = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    out = Path("runs") / datetime.now().strftime("pleural_effusion_fixed30_%Y%m%d_%H%M%S_%f")
    out.mkdir(exist_ok=False)
    print(f"Training on {torch.cuda.get_device_name()} in {out}", flush=True)
    history = []
    head.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(features.cuda()), targets.cuda())
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train)})
    write_csv(out / "history.csv", history)
    config = {"epoch": EPOCHS, "seed": SEED, "labels": LABELS, "backbone_dir": str(BACKBONE),
              "frozen_backbone": True, "max_text_length": 128, "batch_size": 16,
              "head_dimensions": [2048, 128, 2], "dropout": 0.2, "optimizer": "AdamW",
              "learning_rate": 1e-3, "weight_decay": 1e-4, "device": torch.cuda.get_device_name(),
              "feature_order": ["image", "text", "abs(image-text)", "image*text"],
              "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (TRAIN, VALIDATION, REFERENCE)}}
    torch.save({"classifier_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                "config": config}, out / "final_classifier.pt")

    val_x = features_for(val, val_paths, processor, backbone, text_vectors)
    head.eval()
    with torch.no_grad():
        probabilities = head(val_x.cuda()).softmax(dim=1).cpu()
    if not torch.isfinite(probabilities).all():
        raise ValueError("Nonfinite validation probability")
    label_names = {index: label for label, index in LABELS.items()}
    comparison = []
    for row, probs in zip(val, probabilities):
        old = reference[pair_key(row)]
        comparison.append({**row, "original_predicted": old["predicted"],
                           "original_p_supported": old["p_supported"],
                           "new_predicted": label_names[probs.argmax().item()],
                           "new_p_supported": probs[LABELS["Supported"]].item()})
    write_csv(out / "validation_predictions.csv", comparison)
    print(f"Saved epoch {EPOCHS} and {len(comparison)} matched predictions in {out}", flush=True)


if __name__ == "__main__":
    main()
