from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data/metadata/metadata_daugman_strict.csv"


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

    print("=== STRICT METADATA AUDIT ===")
    print(f"Immagini totali: {len(df)}")
    print(f"Famiglie:        {df['family_id'].nunique()}")
    print(f"Soggetti:        {df['subject_id'].nunique()}")
    print(f"Iridi:           {df['iris_id'].nunique()}")
    print()

    print("=== Immagini per iris_id ===")
    iris_counts = df.groupby("iris_id").size().sort_values()
    print(iris_counts.describe().to_string())
    print()

    print("Iris con almeno 2 immagini:", int((iris_counts >= 2).sum()))
    print("Iris con 1 sola immagine:  ", int((iris_counts == 1).sum()))
    print()

    print("=== Famiglie complete ===")
    # Una famiglia completa ha almeno una immagine per 1L,1R,2L,2R
    df["folder_class"] = df["twin_id"] + df["eye"]
    family_classes = df.groupby("family_id")["folder_class"].nunique()
    print("Famiglie con tutte e 4 le classi:", int((family_classes == 4).sum()))
    print("Famiglie incomplete:", int((family_classes < 4).sum()))
    print()

    print("=== Distribuzione folder_class ===")
    print(df["folder_class"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()