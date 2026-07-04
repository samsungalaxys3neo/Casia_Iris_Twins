from pathlib import Path
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "metadata" / "metadata_daugman_strict.csv"

DEFAULT_OUTPUT_MIN2 = PROJECT_ROOT / "data" / "metadata" / "metadata_daugman_strict_min2.csv"
DEFAULT_OUTPUT_COMPLETE = PROJECT_ROOT / "data" / "metadata" / "metadata_daugman_strict_complete_families.csv"
DEFAULT_OUTPUT_SUMMARY = PROJECT_ROOT / "data" / "metadata" / "strict_subsets_summary.txt"


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


def safe_value_counts(series: pd.Series) -> str:
    if len(series) == 0:
        return "Nessun dato."
    return series.value_counts().sort_index().to_string()


def run_subsets(input_path: Path, output_min2: Path, output_complete: Path, output_summary: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata non trovato: {input_path}")

    df = read_metadata(input_path)

    required_cols = ["family_id", "twin_id", "eye", "subject_id", "iris_id"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Colonne mancanti nel metadata: {missing}")

    df["folder_class"] = df["twin_id"].astype(str) + df["eye"].astype(str)

    # Subset 1: solo iris_id con almeno 2 immagini.
    iris_counts = df.groupby("iris_id").size()
    valid_iris_ids = iris_counts[iris_counts >= 2].index
    df_min2 = df[df["iris_id"].isin(valid_iris_ids)].copy()

    # Subset 2: solo famiglie con tutte le classi 1L, 1R, 2L, 2R presenti.
    expected_classes = {"1L", "1R", "2L", "2R"}

    family_classes = df.groupby("family_id")["folder_class"].apply(lambda s: set(s.dropna()))
    complete_family_ids = family_classes[
        family_classes.apply(lambda classes: expected_classes.issubset(classes))
    ].index

    df_complete = df[df["family_id"].isin(complete_family_ids)].copy()

    output_min2.parent.mkdir(parents=True, exist_ok=True)
    output_complete.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    df_min2.to_csv(output_min2, index=False)
    df_complete.to_csv(output_complete, index=False)

    lines = []
    lines.append("=== STRICT SUBSETS SUMMARY ===")
    lines.append("")
    lines.append(f"Input:                {project_display_path(input_path)}")
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
    lines.append(safe_value_counts(df_min2["folder_class"]))
    lines.append("")
    lines.append("=== Subset COMPLETE FAMILIES ===")
    lines.append(f"Immagini: {len(df_complete)}")
    lines.append(f"Famiglie: {df_complete['family_id'].nunique()}")
    lines.append(f"Soggetti: {df_complete['subject_id'].nunique()}")
    lines.append(f"Iridi:    {df_complete['iris_id'].nunique()}")
    lines.append("")
    lines.append("Distribuzione folder_class COMPLETE:")
    lines.append(safe_value_counts(df_complete["folder_class"]))
    lines.append("")
    lines.append("=== Outputs ===")
    lines.append(f"MIN2:              {project_display_path(output_min2)}")
    lines.append(f"COMPLETE FAMILIES: {project_display_path(output_complete)}")
    lines.append(f"Summary:           {project_display_path(output_summary)}")

    output_summary.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path a metadata_daugman_strict.csv.",
    )
    parser.add_argument(
        "--output-min2",
        type=Path,
        default=DEFAULT_OUTPUT_MIN2,
        help="Output con soli iris_id con almeno 2 immagini.",
    )
    parser.add_argument(
        "--output-complete",
        type=Path,
        default=DEFAULT_OUTPUT_COMPLETE,
        help="Output con sole famiglie complete 1L, 1R, 2L, 2R.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help="Output summary txt.",
    )

    args = parser.parse_args()

    run_subsets(
        input_path=resolve_from_project(args.input),
        output_min2=resolve_from_project(args.output_min2),
        output_complete=resolve_from_project(args.output_complete),
        output_summary=resolve_from_project(args.output_summary),
    )


if __name__ == "__main__":
    main()
