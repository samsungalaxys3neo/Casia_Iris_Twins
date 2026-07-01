from pathlib import Path
import argparse
import pandas as pd


# Project root = repository root, assuming this file is inside the scripts/ folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "duplicate_report.csv"


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
    Return a clean path for reports and terminal output.
    If the path is inside the project, show it relative to PROJECT_ROOT.
    Otherwise, show a generic external marker.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return "<external_path>"


def read_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Read metadata while preserving important identifier columns as strings.
    This avoids losing leading zeros in family_id, image_idx, etc.
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

    return pd.read_csv(metadata_path, dtype=dtype_map)


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find exact duplicate files based on SHA-256 hashes.
    Rows with missing SHA-256 are ignored.
    """
    if "sha256" not in df.columns:
        raise ValueError("Colonna 'sha256' non trovata nel metadata.")

    duplicates = df[
        df["sha256"].notna() & df["sha256"].duplicated(keep=False)
    ].copy()

    sort_cols = [
        col for col in ["sha256", "family_id", "twin_id", "eye", "filename"]
        if col in duplicates.columns
    ]

    if sort_cols:
        duplicates = duplicates.sort_values(sort_cols)

    return duplicates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path a metadata.csv. Default: data/metadata/metadata.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path dove salvare duplicate_report.csv. Default: data/metadata/duplicate_report.csv",
    )

    args = parser.parse_args()

    metadata_path = resolve_from_project(args.metadata)
    output_path = resolve_from_project(args.output)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata non trovato: {metadata_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_metadata(metadata_path)
    duplicates = find_duplicates(df)

    if duplicates.empty:
        duplicates.to_csv(output_path, index=False)
        print("Nessun duplicato esatto trovato.")
        print(f"Report vuoto salvato in: {project_display_path(output_path)}")
        return

    duplicates.to_csv(output_path, index=False)

    print(f"Trovate {len(duplicates)} righe coinvolte in duplicati esatti.")
    print(f"Report salvato in: {project_display_path(output_path)}")
    print()

    cols = [
        "sha256",
        "family_id",
        "twin_id",
        "eye",
        "subject_id",
        "iris_id",
        "filename",
        "relative_path",
        "project_relative_path",
    ]
    cols = [col for col in cols if col in duplicates.columns]

    print(duplicates[cols].to_string(index=False))

    print("\n=== Analisi per gruppo duplicato ===")
    for sha, group in duplicates.groupby("sha256"):
        print("\nHash:", sha)
        print("Numero copie:", len(group))
        print("iris_id coinvolti:", sorted(group["iris_id"].dropna().unique()))
        print("family_id coinvolti:", sorted(group["family_id"].dropna().astype(str).unique()))

        detail_cols = [col for col in ["relative_path", "project_relative_path", "iris_id"] if col in group.columns]
        print(group[detail_cols].to_string(index=False))


if __name__ == "__main__":
    main()