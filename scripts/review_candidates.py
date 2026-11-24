
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/padchest_findings_flagged.csv")

df = pd.read_csv(INPUT)

# Select claims not flagged by our rules
candidates = df[df["needs_review"] == False].copy()

print("=== CANDIDATE CLAIM REVIEW ===")
print("Total candidates:", len(candidates))

print("\nCandidates by abnormal status:")
print(candidates["abnormal"].value_counts())

# Randomly select 15 abnormal and 15 normal findings
samples = []

for status in [True, False]:
    subset = candidates[candidates["abnormal"] == status]

    selected = subset.sample(
        n=min(15, len(subset)),
        random_state=42
    )

    samples.append(selected)

review_df = pd.concat(samples)

print("\n=== 30 RANDOM CLAIMS ===")

for i, (_, row) in enumerate(review_df.iterrows(), start=1):
    print(f"\n{i}. {row['claim']}")
    print(f"   Abnormal: {row['abnormal']}")
    print(f"   Split: {row['split']}")

# Save review sample
OUTPUT = Path("data/processed/claim_review_sample.csv")
review_df.to_csv(OUTPUT, index=False)

print("\nReview sample saved to:", OUTPUT)
