from pathlib import Path
import argparse
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEGMENTATION_REPORT = PROJECT_ROOT / "data" / "segmentation_v2" / "segmentation_v2_report.csv"

DEFAULT_OUTPUT_STRICT = PROJECT_ROOT / "data" / "metadata" / "metadata_segmented_strict.csv"
DEFAULT_OUTPUT_REJECTED = PROJECT_ROOT / "data" / "metadata" / "metadata_segmented_strict_rejected.csv"

DEFAULT_OVERLAY_DIR = PROJECT_ROOT / "data" / "segmentation_v2" / "overlays"
DEFAULT_CONTACT_DIR = PROJECT_ROOT / "data" / "segmentation_v2" / "strict_contact_sheets"

THUMB_SIZE = (180, 135)
LABEL_HEIGHT = 42
COLS = 5


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


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def safe_name(relative_path: str) -> str:
    return str(relative_path).replace("/", "__").replace("\\", "__")


def read_segmentation_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
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

    numeric_cols = [
        "iris_center_offset",
        "iris_pupil_radius_ratio",
        "pupil_r",
        "iris_r",
        "iris_score",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_contact_sheet(df, title, output_path, overlay_dir, n=50):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()

    if len(df) > n:
        df = df.sample(n=n, random_state=42).copy()

    thumbs = []

    for _, row in df.iterrows():
        overlay_name = safe_name(row["relative_path"]) + ".png"
        overlay_path = overlay_dir / overlay_name

        if not overlay_path.exists():
            continue

        img = Image.open(overlay_path).convert("RGB")
        img.thumbnail(THUMB_SIZE)

        canvas = Image.new("RGB", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), "white")
        x = (THUMB_SIZE[0] - img.width) // 2
        canvas.paste(img, (x, 0))

        draw = ImageDraw.Draw(canvas)

        ratio = safe_float(row.get("iris_pupil_radius_ratio", "nan"))
        offset = safe_float(row.get("iris_center_offset", "nan"))
        score = safe_float(row.get("iris_score", "nan"))

        label1 = str(row["relative_path"])[:32]
        label2 = f"ratio={ratio:.2f} off={offset:.1f} score={score:.1f}"

        draw.text((4, THUMB_SIZE[1] + 3), label1, fill="black", font=font)
        draw.text((4, THUMB_SIZE[1] + 20), label2, fill="black", font=font)

        thumbs.append(canvas)

    if not thumbs:
        print(f"Nessuna immagine per {title}")
        return

    rows = math.ceil(len(thumbs) / COLS)
    sheet = Image.new(
        "RGB",
        (COLS * THUMB_SIZE[0], rows * (THUMB_SIZE[1] + LABEL_HEIGHT) + 35),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black", font=font)

    y0 = 35

    for i, thumb in enumerate(thumbs):
        x = (i % COLS) * THUMB_SIZE[0]
        y = y0 + (i // COLS) * (THUMB_SIZE[1] + LABEL_HEIGHT)
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)
    print(f"Salvata contact sheet: {project_display_path(output_path)}")


def run_filter(segmentation_report, output_strict, output_rejected, overlay_dir, contact_dir):
    if not segmentation_report.exists():
        raise FileNotFoundError(f"Non trovo il report di segmentazione: {segmentation_report}")

    df = read_segmentation_report(segmentation_report)

    strict_mask = (
        (df["segmentation_ok"] == True)
        & (df["iris_confidence"] == "high")
        & (df["iris_center_offset"] <= 15)
        & (df["iris_pupil_radius_ratio"] >= 2.0)
        & (df["iris_pupil_radius_ratio"] <= 3.1)
        & (df["pupil_r"] >= 25)
        & (df["pupil_r"] <= 75)
        & (df["iris_r"] >= 85)
        & (df["iris_r"] <= 160)
        & (df["iris_score"] >= 60)
    )

    df_strict = df[strict_mask].copy()
    df_rejected = df[~strict_mask].copy()

    output_strict.parent.mkdir(parents=True, exist_ok=True)
    output_rejected.parent.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    df_strict.to_csv(output_strict, index=False)
    df_rejected.to_csv(output_rejected, index=False)

    print("=== STRICT SEGMENTATION FILTER ===")
    print(f"Input report:          {project_display_path(segmentation_report)}")
    print(f"Overlay dir:           {project_display_path(overlay_dir)}")
    print(f"Input totale:          {len(df)}")
    print(f"Accettate strict:      {len(df_strict)}")
    print(f"Rifiutate strict:      {len(df_rejected)}")
    print(f"Percentuale accettata: {len(df_strict) / len(df):.4f}")
    print()

    print("=== Distribuzione per folder_class strict ===")
    if len(df_strict) > 0:
        df_strict["folder_class"] = df_strict["twin_id"] + df_strict["eye"]
        print(df_strict["folder_class"].value_counts().sort_index().to_string())
    else:
        print("Nessuna immagine strict.")
    print()

    print("=== Famiglie rappresentate ===")
    print(f"Famiglie strict: {df_strict['family_id'].nunique()}")
    print()

    print("=== Statistiche strict ===")
    for col in ["pupil_r", "iris_r", "iris_pupil_radius_ratio", "iris_center_offset", "iris_score"]:
        print(col)
        if len(df_strict) > 0:
            print(df_strict[col].describe().to_string())
        else:
            print("Nessun dato.")
        print()

    make_contact_sheet(
        df_strict,
        "STRICT accepted segmentation examples",
        contact_dir / "strict_accepted_examples.png",
        overlay_dir=overlay_dir,
        n=50,
    )

    make_contact_sheet(
        df_rejected[df_rejected["segmentation_ok"] == True],
        "Rejected despite segmentation_ok examples",
        contact_dir / "strict_rejected_but_ok_examples.png",
        overlay_dir=overlay_dir,
        n=50,
    )

    print(f"Salvato strict metadata in:   {project_display_path(output_strict)}")
    print(f"Salvato rejected metadata in: {project_display_path(output_rejected)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmentation-report",
        type=Path,
        default=DEFAULT_SEGMENTATION_REPORT,
        help="Path al segmentation_v2_report.csv.",
    )
    parser.add_argument(
        "--output-strict",
        type=Path,
        default=DEFAULT_OUTPUT_STRICT,
        help="Output metadata_segmented_strict.csv.",
    )
    parser.add_argument(
        "--output-rejected",
        type=Path,
        default=DEFAULT_OUTPUT_REJECTED,
        help="Output metadata_segmented_strict_rejected.csv.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=DEFAULT_OVERLAY_DIR,
        help="Cartella overlay segmentation_v2.",
    )
    parser.add_argument(
        "--contact-dir",
        type=Path,
        default=DEFAULT_CONTACT_DIR,
        help="Cartella contact sheets strict.",
    )

    args = parser.parse_args()

    run_filter(
        segmentation_report=resolve_from_project(args.segmentation_report),
        output_strict=resolve_from_project(args.output_strict),
        output_rejected=resolve_from_project(args.output_rejected),
        overlay_dir=resolve_from_project(args.overlay_dir),
        contact_dir=resolve_from_project(args.contact_dir),
    )


if __name__ == "__main__":
    main()
