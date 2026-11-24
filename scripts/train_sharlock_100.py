"""First SHARLOCK baseline: frozen MedCLIP-ResNet50 + binary classifier.

Run from the SHARLOCK project: python scripts/train_sharlock_100.py
Uses existing splits and provisional labels. This is classifier training,
not backbone fine-tuning. Requires the already installed MedCLIP environment.
"""

import argparse
import copy
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path


LABELS = {"Unsupported": 0, "Supported": 1}
SPLITS = ("train", "validation", "test")


def read_rows(csv_path, image_dir):
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "claim", "verification_label", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty.")
    assigned = {}
    observed = {}
    for row in rows:
        image_id = row["image_id"]
        if not image_id or Path(image_id).name != image_id:
            raise ValueError(f"Expected an image filename: {image_id!r}")
        if not row["claim"].strip():
            raise ValueError(f"Empty claim for {image_id}")
        if row["verification_label"] not in LABELS or row["split"] not in SPLITS:
            raise ValueError(f"Unexpected label or split: {row}")
        if not (image_dir / image_id).is_file():
            raise FileNotFoundError(image_dir / image_id)
        previous = assigned.setdefault(image_id, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Image occurs in multiple splits: {image_id}")
        key = (image_id, row["claim"].strip())
        if key in observed:
            raise ValueError(f"Duplicate image/claim pair, possibly conflicting: {key}")
        observed[key] = row["verification_label"]
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            raise ValueError(f"Empty {split} split.")
        if split == "train" and {r["verification_label"] for r in selected} != set(LABELS):
            raise ValueError("Training requires both labels.")
    return rows


def load_image(path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as original:
        pixels = np.array(original)
        image = original.copy()
    if not np.isfinite(pixels).all() or pixels.max() == pixels.min():
        raise ValueError(f"Invalid or constant image: {path}")
    # Same high-bit-depth scaling as the successful GPU smoke test.
    if pixels.max() > 255:
        pixels = pixels.astype(np.float32)
        pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255
        image = Image.fromarray(pixels.astype(np.uint8))
    return image.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/subset_100"))
    parser.add_argument("--backbone-dir", type=Path, default=Path("pretrained/medclip-resnet"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience and batch-size must be positive")

    csv_path = args.data_dir / "sharlock_binary_100.csv"
    image_dir = args.data_dir / "images"
    rows = read_rows(csv_path, image_dir)
    if not args.backbone_dir.is_dir():
        raise FileNotFoundError(f"Medical checkpoint directory missing: {args.backbone_dir}")

    import numpy as np
    import torch
    import torch.nn as nn
    import transformers
    from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPProcessor
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the SHARLOCK virtual environment.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        counts = {label: sum(r["verification_label"] == label for r in selected) for label in LABELS}
        print(f"{split}: {len({r['image_id'] for r in selected})} images, "
              f"{len(selected)} pairs, {counts}", flush=True)
    print("Split checks passed at IMAGE level; patient separation relies on the original splits.")

    processor = MedCLIPProcessor()
    backbone = MedCLIPModel(vision_cls=MedCLIPVisionModel)
    backbone.from_pretrained(input_dir=str(args.backbone_dir))
    backbone.requires_grad_(False)
    backbone = backbone.to(device).eval()

    # Features are computed once without fitting anything on validation/test.
    # Fixed-length padding makes a claim's representation independent of its batch.
    max_length = 128
    image_cache, text_cache = {}, {}
    with torch.no_grad():
        for image_id in tqdm(sorted({r["image_id"] for r in rows}), desc="Encoding X-rays"):
            inputs = processor(images=load_image(image_dir / image_id), return_tensors="pt")
            vector = backbone.encode_image(inputs["pixel_values"].to(device))
            if not torch.isfinite(vector).all().item():
                raise ValueError(f"Invalid image features: {image_id}")
            image_cache[image_id] = vector[0].cpu()
        claims = sorted({r["claim"] for r in rows})
        for start in tqdm(range(0, len(claims), 4), desc="Encoding claims"):
            batch = claims[start:start + 4]
            # Reject silent loss of qualifiers at the end of a long claim.
            lengths = processor.tokenizer(batch, truncation=False)["input_ids"]
            if any(len(ids) > max_length for ids in lengths):
                raise ValueError("A claim exceeds 128 tokens. Inspect it before training.")
            tokens = processor.tokenizer(batch, padding="max_length", max_length=max_length,
                                         truncation=False, return_tensors="pt")
            vectors = backbone.encode_text(tokens["input_ids"].to(device),
                                           tokens["attention_mask"].to(device))
            if not torch.isfinite(vectors).all().item():
                raise ValueError("Invalid text features.")
            for claim, vector in zip(batch, vectors.cpu()):
                text_cache[claim] = vector.clone()
    del backbone
    torch.cuda.empty_cache()

    datasets = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        visual = torch.stack([image_cache[r["image_id"]] for r in selected])
        textual = torch.stack([text_cache[r["claim"]] for r in selected])
        features = torch.cat([visual, textual, torch.abs(visual - textual), visual * textual], dim=1)
        targets = torch.tensor([LABELS[r["verification_label"]] for r in selected], dtype=torch.long)
        datasets[split] = TensorDataset(features, targets)

    # Both 512-dimensional embeddings plus explicit comparison features.
    head = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(128, 2)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))

    def evaluate(split):
        head.eval()
        features, targets = datasets[split].tensors
        with torch.no_grad():
            logits = head(features.to(device))
            loss = loss_fn(logits, targets.to(device)).item()
            probabilities = logits.softmax(dim=1).cpu()
        predicted = probabilities.argmax(dim=1).numpy()
        truth = targets.numpy()
        metrics = {
            "loss": loss,
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        }
        return metrics, probabilities

    print("\nTraining classifier only. Backbone stays frozen.", flush=True)
    best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(features), targets)
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is not finite.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        validation, _ = evaluate("validation")
        train_loss = total_loss / len(datasets["train"])
        history.append({"epoch": epoch, "train_loss": train_loss, **{"val_" + k: validation[k] for k in ("loss", "accuracy", "macro_f1")}})
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
              f"val_loss={validation['loss']:.4f} val_acc={validation['accuracy']:.3f} "
              f"val_macro_f1={validation['macro_f1']:.3f}", flush=True)
        if validation["loss"] < best_loss:
            best_loss, best_epoch, stale = validation["loss"], epoch, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    head.load_state_dict(best_state)
    metrics = {split: evaluate(split)[0] for split in ("train", "validation")}
    metrics["test"], test_probabilities = evaluate("test")
    run_dir = args.output_dir / ("sharlock_100_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "stage": "frozen_backbone_classifier_training", "backbone": "MedCLIP-ResNet50",
        "backbone_dir": str(args.backbone_dir.resolve()), "labels": LABELS,
        "feature_order": ["image", "text", "abs(image-text)", "image*text"],
        "head_dimensions": [2048, 128, 2], "dropout": 0.2, "max_text_length": max_length,
        "image_preprocessing": "If max>255: per-image min-max to uint8; RGB; MedCLIPProcessor",
        "seed": args.seed, "best_epoch": best_epoch, "learning_rate": 1e-3,
        "weight_decay": 1e-4, "batch_size": args.batch_size, "patience": args.patience,
        "max_epochs": args.epochs, "csv_path": str(csv_path.resolve()),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "confusion_matrix_order": ["Unsupported", "Supported"],
        "note": "Provisional labels; small pipeline experiment, not clinical validation.",
    }
    torch.save({"classifier_state_dict": {k: v.cpu() for k, v in best_state.items()},
                "config": config}, run_dir / "best_classifier.pt")
    (run_dir / "metrics.json").write_text(json.dumps({"config": config, "metrics": metrics}, indent=2))
    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    test_rows = [r.copy() for r in rows if r["split"] == "test"]
    for row, probabilities in zip(test_rows, test_probabilities):
        row["predicted_label"] = list(LABELS)[probabilities.argmax().item()]
        row["supported_probability"] = probabilities[1].item()
    with (run_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"\nBest epoch: {best_epoch}")
    print("TEST RESULTS:", json.dumps(metrics["test"], indent=2))
    print("Confusion matrix: rows=true, columns=predicted; order=[Unsupported, Supported]")
    print("Saved to:", run_dir.resolve())


if __name__ == "__main__":
    main()
