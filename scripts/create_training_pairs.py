
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/padchest_findings_raw.csv")
OUTPUT = Path("data/processed/sharlock_binary_v1.csv")

df = pd.read_csv(INPUT)

# Simple, explicit clinical contradictions
positive_to_negative = {
    "Cardiomegaly.": "No cardiomegaly is present.",
    "Aortic elongation.": "No aortic elongation is present.",
    "Dorsal scoliosis.": "No dorsal scoliosis is present.",
    "Scoliosis.": "No scoliosis is present.",
    "Pleural effusion.": "No pleural effusion is present.",
    "Left pleural effusion.": "No left pleural effusion is present.",
    "Right pleural effusion.": "No right pleural effusion is present.",
    "Aortic calcification.": "No aortic calcification is present.",
    "Hiatal hernia.": "No hiatal hernia is present.",
    "Biapical pleural thickening.": "No biapical pleural thickening is present.",
    "Bilateral apical pleural thickening.": "No bilateral apical pleural thickening is present.",
    "Air trapping.": "No air trapping is present."
}

negative_to_positive = {
    "No pleural effusion.": "Pleural effusion is present.",
    "No pleural effusion is appreciated.": "Pleural effusion is present.",
    "There is no evidence of effusion.": "Pleural effusion is present.",
    "No signs of pneumothorax.": "Pneumothorax is present.",
    "No consolidations are observed.": "Pulmonary consolidation is present.",
    "No consolidations are appreciated.": "Pulmonary consolidation is present.",
    "No pulmonary infiltrates are appreciated.": "Pulmonary infiltrates are present."
}

records = []

for _, row in df.iterrows():
    original = str(row["claim"]).strip()

    if original in positive_to_negative:
        opposite = positive_to_negative[original]
    elif original in negative_to_positive:
        opposite = negative_to_positive[original]
    else:
        continue

    # Original statement: candidate Supported
    records.append({
        "image_id": row["image_id"],
        "claim": original,
        "verification_label": "Supported",
        "label_origin": "report_derived",
        "split": row["split"]
    })

    # Contradictory statement: candidate Unsupported
    records.append({
        "image_id": row["image_id"],
        "claim": opposite,
        "verification_label": "Unsupported",
        "label_origin": "synthetic_contradiction",
        "split": row["split"]
    })

result = pd.DataFrame(records)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("=== SHARLOCK BINARY DATASET V1 ===")
print("Total training examples:", len(result))

print("\nLabels:")
print(result["verification_label"].value_counts())

print("\nExamples by split:")
print(result["split"].value_counts())

print("\nUnique images:", result["image_id"].nunique())
print("\nSaved to:", OUTPUT)
