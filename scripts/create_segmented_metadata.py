from pathlib import Path
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEGMENTATION_REPORT = PROJECT_ROOT / "data" / "segmentation_v2" / "segmentation_v2_report.csv"

DEFAULT_OUTPUT_HIGH = PROJECT_ROOT / "data" / "metadata" / "metadata_segmented_high.csv"
DEFAULT_OUTPUT_HIGH_MEDIUM = PROJECT_ROOT / "data" / "metadata" / "metadata_segmented_high_medium.csv"
DEFAULT_OUTPUT_REJECTED = PROJECT_ROOT / "data" / "metadata" / "metadata_segmentation_rejected.csv"


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


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmentation-report",
        type=Path,
        default=DEFAULT_SEGMENTATION_REPORT,
        help="Path al segmentation_v2_report.csv.",
    )
    parser.add_argument(
        "--output-high",
        type=Path,
        default=DEFAULT_OUTPUT_HIGH,
        help="Output metadata solo high confidence.",
    )
    parser.add_argument(
        "--output-high-medium",
        type=Path,
        default=DEFAULT_OUTPUT_HIGH_MEDIUM,
        help="Output metadata high + medium confidence.",
    )
    parser.add_argument(
        "--output-rejected",
        type=Path,
        default=DEFAULT_OUTPUT_REJECTED,
        help="Output metadata rejected/check/fail/low/none.",
    )

    args = parser.parse_args()

    segmentation_report = resolve_from_project(args.segmentation_report)
    output_high = resolve_from_project(args.output_high)
    output_high_medium = resolve_from_project(args.output_high_medium)
    output_rejected = resolve_from_project(args.output_rejected)

    if not segmentation_report.exists():
        raise FileNotFoundError(f"Non trovo il report di segmentazione: {segmentation_report}")

    df = pd.read_csv(
        segmentation_report,
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

    df["segmentation_ok"] = as_bool(df["segmentation_ok"])
    df["iris_confidence"] = df["iris_confidence"].fillna("none").astype(str)

    df_high = df[
        (df["segmentation_ok"] == True)
        & (df["iris_confidence"] == "high")
    ].copy()

    df_high_medium = df[
        (df["segmentation_ok"] == True)
        & (df["iris_confidence"].isin(["high", "medium"]))
    ].copy()

    df_rejected = df[
        ~(
            (df["segmentation_ok"] == True)
            & (df["iris_confidence"].isin(["high", "medium"]))
        )
    ].copy()

    output_high.parent.mkdir(parents=True, exist_ok=True)
    output_high_medium.parent.mkdir(parents=True, exist_ok=True)
    output_rejected.parent.mkdir(parents=True, exist_ok=True)

    df_high.to_csv(output_high, index=False)
    df_high_medium.to_csv(output_high_medium, index=False)
    df_rejected.to_csv(output_rejected, index=False)

    print("=== SEGMENTED METADATA REPORT ===")
    print(f"Input report:                         {project_display_path(segmentation_report)}")
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
    if len(df_high) > 0:
        df_high["folder_class"] = df_high["twin_id"] + df_high["eye"]
        print(df_high["folder_class"].value_counts().sort_index().to_string())
    else:
        print("Nessuna immagine.")
    print()

    print("=== Immagini per folder_class nel dataset HIGH+MEDIUM ===")
    if len(df_high_medium) > 0:
        df_high_medium["folder_class"] = df_high_medium["twin_id"] + df_high_medium["eye"]
        print(df_high_medium["folder_class"].value_counts().sort_index().to_string())
    else:
        print("Nessuna immagine.")
    print()

    print("=== Famiglie rappresentate ===")
    print(f"Famiglie in HIGH:        {df_high['family_id'].nunique()}")
    print(f"Famiglie in HIGH+MEDIUM: {df_high_medium['family_id'].nunique()}")
    print()

    print(f"Salvato HIGH in:        {project_display_path(output_high)}")
    print(f"Salvato HIGH+MEDIUM in: {project_display_path(output_high_medium)}")
    print(f"Salvato REJECTED in:    {project_display_path(output_rejected)}")


if __name__ == "__main__":
    main()
