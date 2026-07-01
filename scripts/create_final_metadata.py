from pathlib import Path
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUALITY_REPORT = PROJECT_ROOT / "data" / "quality" / "quality_report.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata_final.csv"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=DEFAULT_QUALITY_REPORT,
        help="Path al quality_report.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path di output per metadata_final.csv.",
    )

    args = parser.parse_args()

    quality_report = resolve_from_project(args.quality_report)
    output_path = resolve_from_project(args.output)

    if not quality_report.exists():
        raise FileNotFoundError(f"Quality report non trovato: {quality_report}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        quality_report,
        dtype={
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
        },
    )

    old_absolute_column = "absolute" + "_path"
    if old_absolute_column in df.columns:
        df = df.drop(columns=[old_absolute_column])

    numeric_cols = [
        "focus_score",
        "contrast",
        "bright_ratio",
        "dark_ratio",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Soglie data-driven: prendiamo l'1% più estremo per ogni metrica.
    focus_q01 = df["focus_score"].quantile(0.01)
    contrast_q01 = df["contrast"].quantile(0.01)
    bright_q99 = df["bright_ratio"].quantile(0.99)
    dark_q99 = df["dark_ratio"].quantile(0.99)

    df["flag_low_focus"] = df["focus_score"] <= focus_q01
    df["flag_low_contrast"] = df["contrast"] <= contrast_q01
    df["flag_high_bright_ratio"] = df["bright_ratio"] >= bright_q99
    df["flag_high_dark_ratio"] = df["dark_ratio"] >= dark_q99

    # Per ora non scartiamo nulla: i flag servono solo per ispezione.
    df["quality_reject"] = False

    df["quality_flag_any"] = (
        df["flag_low_focus"]
        | df["flag_low_contrast"]
        | df["flag_high_bright_ratio"]
        | df["flag_high_dark_ratio"]
    )

    df.to_csv(output_path, index=False)

    print("=== FINAL METADATA REPORT ===")
    print(f"Input:          {project_display_path(quality_report)}")
    print(f"Output:         {project_display_path(output_path)}")
    print(f"Input immagini: {len(df)}")
    print()
    print("=== Soglie usate ===")
    print(f"focus_score <= {focus_q01:.4f}  -> flag_low_focus")
    print(f"contrast <= {contrast_q01:.4f}  -> flag_low_contrast")
    print(f"bright_ratio >= {bright_q99:.4f} -> flag_high_bright_ratio")
    print(f"dark_ratio >= {dark_q99:.4f}    -> flag_high_dark_ratio")
    print()
    print("=== Conteggio flag ===")
    print(f"flag_low_focus:          {int(df['flag_low_focus'].sum())}")
    print(f"flag_low_contrast:       {int(df['flag_low_contrast'].sum())}")
    print(f"flag_high_bright_ratio:  {int(df['flag_high_bright_ratio'].sum())}")
    print(f"flag_high_dark_ratio:    {int(df['flag_high_dark_ratio'].sum())}")
    print(f"quality_flag_any:        {int(df['quality_flag_any'].sum())}")
    print(f"quality_reject:          {int(df['quality_reject'].sum())}")


if __name__ == "__main__":
    main()
