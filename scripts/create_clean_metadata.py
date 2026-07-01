from pathlib import Path
import argparse
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata_clean.csv"
DEFAULT_REMOVED_PATH = PROJECT_ROOT / "data" / "metadata" / "removed_duplicates.csv"


def resolve_from_project(path: Path) -> Path:
    """
    Resolve a path in a portable way.
    If the path is relative, interpret it relative to the project root.
    If the path is absolute, keep it as it is.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_display_path(path: Path) -> str:
    """
    Return a clean path for terminal output.
    If the path is inside the project, show it relative to PROJECT_ROOT.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return "<external_path>"


def read_metadata(input_path: Path) -> pd.DataFrame:
    """
    Read metadata while preserving identifier columns as strings.
    This avoids losing leading zeros in family_id or image_idx.
    """
    dtype_map = {
        "family_id": str,
        "twin_id": str,
        "eye": str,
        "subject_id": str,
        "iris_id": str,
        "image_idx": str,
        "filename": str,
        "relative_path": str,
        "project_relative_path": str,
        "sha256": str,
    }

    return pd.read_csv(input_path, dtype=dtype_map)


def create_clean_metadata(
    input_path: Path,
    output_path: Path,
    removed_path: Path,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Metadata non trovato: {input_path}")

    df = read_metadata(input_path)

    if "sha256" not in df.columns:
        raise ValueError("Colonna 'sha256' non trovata nel metadata.")

    sort_cols = ["family_id", "twin_id", "eye", "image_idx", "filename"]
    missing_sort_cols = [col for col in sort_cols if col not in df.columns]

    if missing_sort_cols:
        raise ValueError(f"Colonne mancanti per ordinamento: {missing_sort_cols}")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    valid_hash = df["sha256"].notna()
    duplicated_rows = df[valid_hash & df["sha256"].duplicated(keep="first")].copy()
    df_clean = df[~(valid_hash & df["sha256"].duplicated(keep="first"))].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(output_path, index=False)
    duplicated_rows.to_csv(removed_path, index=False)

    print("=== CLEAN METADATA REPORT ===")
    print(f"Input originale: {len(df)} immagini")
    print(f"Output pulito:   {len(df_clean)} immagini")
    print(f"Rimosse:         {len(df) - len(df_clean)} immagini")
    print()
    print(f"Salvato metadata pulito in: {project_display_path(output_path)}")
    print(f"Salvato elenco duplicati rimossi in: {project_display_path(removed_path)}")

    if len(duplicated_rows) > 0:
        print("\n=== Duplicati rimossi ===")
        cols = [
            "family_id",
            "twin_id",
            "eye",
            "iris_id",
            "filename",
            "relative_path",
            "sha256",
        ]
        cols = [col for col in cols if col in duplicated_rows.columns]
        print(duplicated_rows[cols].to_string(index=False))

    return df_clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path a metadata.csv. Default: data/metadata/metadata.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path dove salvare metadata_clean.csv. Default: data/metadata/metadata_clean.csv",
    )
    parser.add_argument(
        "--removed",
        type=Path,
        default=DEFAULT_REMOVED_PATH,
        help="Path dove salvare removed_duplicates.csv. Default: data/metadata/removed_duplicates.csv",
    )

    args = parser.parse_args()

    input_path = resolve_from_project(args.input)
    output_path = resolve_from_project(args.output)
    removed_path = resolve_from_project(args.removed)

    create_clean_metadata(input_path, output_path, removed_path)


if __name__ == "__main__":
    main()
