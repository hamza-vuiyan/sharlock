
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/padchest_findings_raw.csv")

df = pd.read_csv(DATA_PATH)
claims = df["claim"].fillna("").astype(str)

patterns = {
    "negation": r"\b(no|without|absent|absence|not|free of)\b",

    "uncertainty": (
        r"\b(possible|possibly|probable|probably|"
        r"may|might|suggestive|suspected|likely)\b"
    ),

    "temporal": (
        r"\b(previous|prior|increased|decreased|"
        r"stable|unchanged|new|worsening|improving|"
        r"resolved|compared|persistent)\b"
    ),

    "generic_normal": (
        r"(?i)^(no (other )?(significant|relevant|"
        r"remarkable) (findings|alterations)|"
        r"no significant radiological alterations|"
        r"normal study)\.?$"
    ),
}

print("=== CLAIM QUALITY ANALYSIS ===")
print("Total claims:", len(df))

for category, pattern in patterns.items():
    mask = claims.str.contains(
        pattern, case=False, regex=True
    )

    print(f"\n{category.upper()}: {mask.sum()}")

    print("Examples:")
    for claim in claims[mask].head(5):
        print(" -", claim)

print("\n=== OTHER STATISTICS ===")
print("Unique images:", df["image_id"].nunique())
print("Empty claims:", claims.str.strip().eq("").sum())

print("\nAbnormal values:")
print(df["abnormal"].value_counts(dropna=False))
