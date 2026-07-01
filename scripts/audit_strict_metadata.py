from pathlib import Path
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "metadata" / "metadata_daugman_strict.csv"


def resolve_from_project(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return "<external_path>"


def read_metadata(path: Path) -> pd.DataFrame:
    dtype_map = {
        "family_id": str,
        "twin_id": str,
        "eye": str,
        "image_idx": str,
        "subject_id": str,
        "iris_id": str,
        "filename": str,
        "relative_path": str,
        "project_relative_path": str,
        "sha256": str,
    }

    df = pd.read_csv(path, dtype=dtype_map)

    old_absolute_column = "absolute" + "_path"
    if old_absolute_column in df.columns:
        df = df.drop(columns=[old_absolute_column])

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path a metadata_daugman_strict.csv.",
    )

    args = parser.parse_args()
    input_path = resolve_from_project(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata non trovato: {input_path}")

    df = read_metadata(input_path)

    required_cols = ["family_id", "twin_id", "eye", "subject_id", "iris_id"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Colonne mancanti nel metadata: {missing}")

    df["folder_class"] = df["twin_id"].astype(str) + df["eye"].astype(str)

    print("=== STRICT METADATA AUDIT ===")
    print(f"Input:           {project_display_path(input_path)}")
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
    expected_classes = {"1L", "1R", "2L", "2R"}

    family_classes = df.groupby("family_id")["folder_class"].apply(lambda s: set(s.dropna()))
    complete_mask = family_classes.apply(lambda classes: expected_classes.issubset(classes))

    print("Famiglie con tutte e 4 le classi:", int(complete_mask.sum()))
    print("Famiglie incomplete:", int((~complete_mask).sum()))
    print()

    print("=== Distribuzione folder_class ===")
    print(df["folder_class"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
