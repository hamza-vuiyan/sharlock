
import pandas as pd
import re
from pathlib import Path

INPUT = Path("data/processed/padchest_findings_raw.csv")
OUTPUT = Path("data/processed/padchest_findings_flagged.csv")

df = pd.read_csv(INPUT)

patterns = {
    "uncertainty": (
        r"\b(?:possible|possibly|probable|probably|"
        r"may|might|suggestive|suspected|likely|"
        r"cannot exclude|questionable)\b"
    ),

    "temporal": (
        r"\b(?:previous|prior|compared|since|interval|"
        r"persistent|persisting|unchanged|stable|"
        r"worsening|improving|resolved|newly)\b|"
        r"\b(?:has increased|has decreased)\b"
    ),

    "generic_normal": (
        r"\b(?:no (?:other )?(?:significant|relevant) "
        r"(?:findings|alterations)|"
        r"no relevant findings|"
        r"no radiological findings of interest|"
        r"normal study|"
        r"no significant radiological alterations)\b"
    ),

    "incomplete_fragment": (
        r"^(?:with formation|suggestive of|likely of|"
        r"in relation to|compatible with|consistent with)\b"
    ),

    "compound_sentence": r"\b(?:and|or)\b",
}

def get_review_reasons(row):
    claim = str(row["claim"]).strip()
    reasons = []

    for category, pattern in patterns.items():
        if re.search(pattern, claim, flags=re.IGNORECASE):
            reasons.append(category)

    if pd.notna(row["progression"]):
        if "temporal" not in reasons:
            reasons.append("temporal")

    return "; ".join(reasons)

df["review_reason"] = df.apply(get_review_reasons, axis=1)

df["needs_review"] = df["review_reason"] != ""

df.to_csv(OUTPUT, index=False)

print("=== CLAIM REVIEW ANALYSIS ===")
print("Total claims:", len(df))

print("Flagged for review:", df["needs_review"].sum())

print("Not flagged:", (~df["needs_review"]).sum())

print("\nReview categories:")

for category in patterns:
    count = df["review_reason"].str.contains(
        category, regex=False
    ).sum()
    print(f"{category}: {count}")

print("\nExample flagged claims:")

for _, row in df[df["needs_review"]].head(10).iterrows():
    print(f"- {row['claim']}")
    print(f"  Reason: {row['review_reason']}")

print("\nSaved to:", OUTPUT)
