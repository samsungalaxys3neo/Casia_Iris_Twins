from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = PROJECT_ROOT / "data/metadata/metadata_segmented_strict.csv"

OUTPUT_MIN2 = PROJECT_ROOT / "data/metadata/metadata_segmented_strict_min2.csv"
OUTPUT_COMPLETE = PROJECT_ROOT / "data/metadata/metadata_segmented_strict_complete_families.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "data/metadata/strict_subsets_summary.txt"


def main():
    df = pd.read_csv(
        INPUT,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    df["folder_class"] = df["twin_id"] + df["eye"]

    # Subset 1: solo iris_id con almeno 2 immagini
    iris_counts = df.groupby("iris_id").size()
    valid_iris_ids = iris_counts[iris_counts >= 2].index
    df_min2 = df[df["iris_id"].isin(valid_iris_ids)].copy()

    # Subset 2: solo famiglie con tutte le classi 1L, 1R, 2L, 2R presenti
    family_classes = df.groupby("family_id")["folder_class"].nunique()
    complete_family_ids = family_classes[family_classes == 4].index
    df_complete = df[df["family_id"].isin(complete_family_ids)].copy()

    df_min2.to_csv(OUTPUT_MIN2, index=False)
    df_complete.to_csv(OUTPUT_COMPLETE, index=False)

    lines = []
    lines.append("=== STRICT SUBSETS SUMMARY ===")
    lines.append("")
    lines.append(f"Input strict immagini: {len(df)}")
    lines.append(f"Input strict famiglie: {df['family_id'].nunique()}")
    lines.append(f"Input strict soggetti: {df['subject_id'].nunique()}")
    lines.append(f"Input strict iridi:    {df['iris_id'].nunique()}")
    lines.append("")
    lines.append("=== Subset MIN2 ===")
    lines.append(f"Immagini: {len(df_min2)}")
    lines.append(f"Famiglie: {df_min2['family_id'].nunique()}")
    lines.append(f"Soggetti: {df_min2['subject_id'].nunique()}")
    lines.append(f"Iridi:    {df_min2['iris_id'].nunique()}")
    lines.append("")
    lines.append("Distribuzione folder_class MIN2:")
    lines.append(df_min2["folder_class"].value_counts().sort_index().to_string())
    lines.append("")
    lines.append("=== Subset COMPLETE FAMILIES ===")
    lines.append(f"Immagini: {len(df_complete)}")
    lines.append(f"Famiglie: {df_complete['family_id'].nunique()}")
    lines.append(f"Soggetti: {df_complete['subject_id'].nunique()}")
    lines.append(f"Iridi:    {df_complete['iris_id'].nunique()}")
    lines.append("")
    lines.append("Distribuzione folder_class COMPLETE:")
    lines.append(df_complete["folder_class"].value_counts().sort_index().to_string())

    OUTPUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print()
    print(f"Salvato MIN2 in:              {OUTPUT_MIN2}")
    print(f"Salvato COMPLETE FAMILIES in: {OUTPUT_COMPLETE}")
    print(f"Salvato summary in:           {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()