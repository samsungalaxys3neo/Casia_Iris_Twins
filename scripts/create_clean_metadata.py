from pathlib import Path
import pandas as pd


INPUT_PATH = Path("iris_twins_project/data/metadata/metadata.csv")
OUTPUT_PATH = Path("iris_twins_project/data/metadata/metadata_clean.csv")
REMOVED_PATH = Path("iris_twins_project/data/metadata/removed_duplicates.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    # Ordiniamo in modo stabile: tiene la prima immagine secondo family/twin/eye/image_idx
    df = df.sort_values(
        ["family_id", "twin_id", "eye", "image_idx", "filename"]
    ).reset_index(drop=True)

    duplicated_rows = df[df["sha256"].duplicated(keep="first")].copy()

    df_clean = df.drop_duplicates(subset=["sha256"], keep="first").copy()

    df_clean.to_csv(OUTPUT_PATH, index=False)
    duplicated_rows.to_csv(REMOVED_PATH, index=False)

    print("=== CLEAN METADATA REPORT ===")
    print(f"Input originale: {len(df)} immagini")
    print(f"Output pulito:   {len(df_clean)} immagini")
    print(f"Rimosse:         {len(df) - len(df_clean)} immagini")
    print()
    print(f"Salvato metadata pulito in: {OUTPUT_PATH}")
    print(f"Salvato elenco duplicati rimossi in: {REMOVED_PATH}")

    if len(duplicated_rows) > 0:
        print("\n=== Duplicati rimossi ===")
        cols = ["family_id", "twin_id", "eye", "iris_id", "filename", "relative_path", "sha256"]
        print(duplicated_rows[cols].to_string(index=False))


if __name__ == "__main__":
    main()