
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
ANNOTATION_DIR = Path("data/annotations")
OUTPUT_DIR = Path("data/subset_100")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared verification pairs
pairs = pd.read_csv(
    DATA_DIR / "sharlock_binary_v1.csv",
    dtype={"image_id": str}
)

# Load original finding statements
findings = pd.read_csv(
    DATA_DIR / "padchest_findings_raw.csv",
    dtype={"image_id": str}
)

# Select 100 unique images using existing splits
split_sizes = {
    "train": 70,
    "validation": 15,
    "test": 15
}

selected_ids = []

for split, n in split_sizes.items():
    images = (
        pairs.loc[pairs["split"] == split, "image_id"]
        .drop_duplicates()
    )

    sampled = images.sample(n=n, random_state=42)
    selected_ids.extend(sampled.tolist())

selected_ids = set(selected_ids)

# Keep all verification pairs for selected images
subset_pairs = pairs[
    pairs["image_id"].isin(selected_ids)
].copy()

# Keep all original findings for selected images
subset_findings = findings[
    findings["image_id"].isin(selected_ids)
].copy()

# Preserve original JSON annotations
with open(
    ANNOTATION_DIR / "grounded_reports_20240819.json"
) as f:
    reports = json.load(f)

subset_reports = [
    study for study in reports
    if study["ImageID"] in selected_ids
]

# Save subset files
subset_pairs.to_csv(
    OUTPUT_DIR / "sharlock_binary_100.csv",
    index=False
)

subset_findings.to_csv(
    OUTPUT_DIR / "findings_100.csv",
    index=False
)

with open(
    OUTPUT_DIR / "grounded_reports_100.json", "w"
) as f:
    json.dump(subset_reports, f, indent=2)

# Save image list
pd.DataFrame({
    "image_id": sorted(selected_ids)
}).to_csv(
    OUTPUT_DIR / "image_list_100.csv",
    index=False
)

print("=== PADChest-GR SUBSET ===")
print("Unique studies:", len(selected_ids))
print("Original findings:", len(subset_findings))
print("Verification pairs:", len(subset_pairs))

print("\nStudies by split:")
print(subset_pairs.groupby("split")["image_id"].nunique())

print("\nVerification labels:")
print(subset_pairs["verification_label"].value_counts())

print("\nSaved to:", OUTPUT_DIR)
