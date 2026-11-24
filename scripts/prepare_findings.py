
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/annotations")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load annotations
with open(DATA_DIR / "grounded_reports_20240819.json") as f:
    reports = json.load(f)

# Load existing dataset splits
master = pd.read_csv(
    DATA_DIR / "master_table.csv",
    usecols=["ImageID", "split"],
    dtype=str
)

# One split per image
split_map = (
    master.drop_duplicates("ImageID")
    .set_index("ImageID")["split"]
    .to_dict()
)

records = []

for study in reports:
    image_id = study["ImageID"]
    split = split_map[image_id]

    for index, finding in enumerate(study["findings"], start=1):
        records.append({
            "claim_id": f"{image_id}_{index}",
            "study_id": study["StudyID"],
            "image_id": image_id,
            "claim": finding["sentence_en"],
            "abnormal": finding["abnormal"],
            "certainty": "",
            "verification_label": "",
            "labels": json.dumps(finding.get("labels", [])),
            "locations": json.dumps(finding.get("locations", [])),
            "progression": finding.get("progression"),
            "boxes": json.dumps(finding.get("boxes", [])),
            "previous_image_id": study.get("PreviousImageID"),
            "split": split
        })

df = pd.DataFrame(records)

output_path = OUTPUT_DIR / "padchest_findings_raw.csv"
df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print("Total findings:", len(df))
print("Unique images:", df["image_id"].nunique())

print("\nFindings by split:")
print(df["split"].value_counts())

print("\nOutput saved to:", output_path)
