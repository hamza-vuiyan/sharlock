
import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/annotations")

# Load PadChest-GR JSON
with open(DATA_DIR / "grounded_reports_20240819.json", "r") as f:
    reports = json.load(f)

print("=== JSON DATASET ===")
print("Total studies:", len(reports))

# Inspect first study
first_study = reports[0]

print("\nFirst study ID:", first_study["StudyID"])
print("Image ID:", first_study["ImageID"])
print("Number of findings:", len(first_study["findings"]))

print("\nClinical findings:")
for finding in first_study["findings"]:
    print("-", finding["sentence_en"])
    print("  Abnormal:", finding["abnormal"])

# Load master CSV
df = pd.read_csv(DATA_DIR / "master_table.csv")

print("\n=== MASTER CSV ===")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))
