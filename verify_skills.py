"""
verify_skills.py

Reads advanced_skills.csv (Hybrid fused skills) and assigns
verification categories based on confidence_score.

Outputs:
    verified_skills.csv  (same folder as advanced_skills.csv by default)
"""

import argparse
from pathlib import Path

import pandas as pd

import config  # uses config.OUTPUT_DIR


# --- thresholds and labels ----------------------------------------------------

def verification_level(conf: float) -> str:
    """
    Map a confidence score in [0,1] to a verification level.
    Adjust thresholds here if needed.
    """
    if pd.isna(conf):
        return "Unknown"

    if conf >= 0.85:
        return "Verified_HIGH"
    elif conf >= 0.65:
        return "Verified_MEDIUM"
    else:
        return "Low_Confidence"


def main():
    parser = argparse.ArgumentParser(
        description="Assign verification categories to advanced skills."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing advanced_skills.csv (default: config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="advanced_skills.csv",
        help="Input skills file name (default: advanced_skills.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="verified_skills.csv",
        help="Output file name (default: verified_skills.csv)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    input_path = output_dir / args.input
    output_path = output_dir / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Reading skills from {input_path}")
    df = pd.read_csv(input_path)

    if "confidence_score" not in df.columns:
        raise ValueError("advanced_skills.csv must contain 'confidence_score' column.")

    # Ensure numeric
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")

    # Add verification category
    df["verification_level"] = df["confidence_score"].apply(verification_level)

    # Convenience boolean: is this considered 'verified'?
    df["is_verified"] = df["verification_level"].isin(
        ["Verified_HIGH", "Verified_MEDIUM"]
    )

    print("[INFO] Verification distribution:")
    print(df["verification_level"].value_counts())

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved verified skills to {output_path}")


if __name__ == "__main__":
    main()
