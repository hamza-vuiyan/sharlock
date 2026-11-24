
import json
import pandas as pd
from pathlib import Path

data_dir = Path("data/annotations")

with open(data_dir / "grounded_reports_20240819.json") as f:
    reports = json.load(f)

df = pd.read_csv(
    data_dir / "master_table.csv",
    dtype={"ImageID": str}
)

json_ids = {study["ImageID"] for study in reports}
csv_ids = set(df["ImageID"].dropna())

print("Unique JSON images:", len(json_ids))
print("Unique CSV images:", len(csv_ids))

missing = json_ids - csv_ids

print("JSON images missing from CSV:", len(missing))

# Check whether an image belongs to multiple splits
split_counts = df.groupby("ImageID")["split"].nunique()

print(
    "Images assigned to multiple splits:",
    (split_counts > 1).sum()
)

print("\nStudies per split:")
print(
    df.drop_duplicates("ImageID")["split"]
    .value_counts()
)
