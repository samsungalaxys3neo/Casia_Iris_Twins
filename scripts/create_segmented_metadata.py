from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEGMENTATION_REPORT = PROJECT_ROOT / "data/segmentation_v2/segmentation_v2_report.csv"

OUTPUT_HIGH = PROJECT_ROOT / "data/metadata/metadata_segmented_high.csv"
OUTPUT_HIGH_MEDIUM = PROJECT_ROOT / "data/metadata/metadata_segmented_high_medium.csv"
OUTPUT_REJECTED = PROJECT_ROOT / "data/metadata/metadata_segmentation_rejected.csv"


def main():
    if not SEGMENTATION_REPORT.exists():
        raise FileNotFoundError(f"Non trovo il report di segmentazione: {SEGMENTATION_REPORT}")

    df = pd.read_csv(
        SEGMENTATION_REPORT,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    # Normalizziamo eventuali valori mancanti
    df["iris_confidence"] = df["iris_confidence"].fillna("none")

    # Dataset principale: solo segmentazioni OK ad alta confidenza
    df_high = df[
        (df["segmentation_ok"] == True)
        & (df["iris_confidence"] == "high")
    ].copy()

    # Dataset secondario: segmentazioni OK high + medium
    df_high_medium = df[
        (df["segmentation_ok"] == True)
        & (df["iris_confidence"].isin(["high", "medium"]))
    ].copy()

    # Casi esclusi
    df_rejected = df[
        ~(
            (df["segmentation_ok"] == True)
            & (df["iris_confidence"].isin(["high", "medium"]))
        )
    ].copy()

    OUTPUT_HIGH.parent.mkdir(parents=True, exist_ok=True)

    df_high.to_csv(OUTPUT_HIGH, index=False)
    df_high_medium.to_csv(OUTPUT_HIGH_MEDIUM, index=False)
    df_rejected.to_csv(OUTPUT_REJECTED, index=False)

    print("=== SEGMENTED METADATA REPORT ===")
    print(f"Input totale:                         {len(df)}")
    print()
    print(f"High confidence:                      {len(df_high)}")
    print(f"High + medium confidence:             {len(df_high_medium)}")
    print(f"Rejected / check / fail / low / none: {len(df_rejected)}")
    print()

    print("=== Distribuzione iris_confidence ===")
    print(df["iris_confidence"].value_counts(dropna=False).to_string())
    print()

    print("=== Distribuzione segmentation_ok ===")
    print(df["segmentation_ok"].value_counts(dropna=False).to_string())
    print()

    print("=== Immagini per folder_class nel dataset HIGH ===")
    df_high["folder_class"] = df_high["twin_id"] + df_high["eye"]
    print(df_high["folder_class"].value_counts().sort_index().to_string())
    print()

    print("=== Immagini per folder_class nel dataset HIGH+MEDIUM ===")
    df_high_medium["folder_class"] = df_high_medium["twin_id"] + df_high_medium["eye"]
    print(df_high_medium["folder_class"].value_counts().sort_index().to_string())
    print()

    print("=== Famiglie rappresentate ===")
    print(f"Famiglie in HIGH:        {df_high['family_id'].nunique()}")
    print(f"Famiglie in HIGH+MEDIUM: {df_high_medium['family_id'].nunique()}")
    print()

    print(f"Salvato HIGH in:        {OUTPUT_HIGH}")
    print(f"Salvato HIGH+MEDIUM in: {OUTPUT_HIGH_MEDIUM}")
    print(f"Salvato REJECTED in:    {OUTPUT_REJECTED}")


if __name__ == "__main__":
    main()