from pathlib import Path
import argparse

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUALITY_REPORT = PROJECT_ROOT / "data" / "quality" / "quality_report.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "CASIA-Iris-Twins"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "quality" / "outliers"

DEFAULT_N_OUTLIERS = 25
THUMB_SIZE = (160, 120)
LABEL_HEIGHT = 42
GRID_COLS = 5


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


def read_quality_report(quality_report: Path) -> pd.DataFrame:
    """
    Read quality report while preserving important identifier columns as strings.
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

    df = pd.read_csv(quality_report, dtype=dtype_map)

    numeric_cols = [
        "brightness",
        "contrast",
        "focus_score",
        "dark_ratio",
        "bright_ratio",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_image_path(row: pd.Series, dataset_root: Path) -> Path:
    """
    Build the image path from the portable relative_path column.
    """
    relative_path = row.get("relative_path", None)

    if pd.notna(relative_path) and str(relative_path).strip():
        image_path = dataset_root / str(relative_path)
        if image_path.exists():
            return image_path

    project_relative_path = row.get("project_relative_path", None)

    if pd.notna(project_relative_path) and str(project_relative_path).strip():
        image_path = PROJECT_ROOT / str(project_relative_path)
        if image_path.exists():
            return image_path

    if pd.notna(relative_path) and str(relative_path).strip():
        return dataset_root / str(relative_path)

    raise ValueError("No relative_path or project_relative_path available for this row.")


def load_font(size=12):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_contact_sheet(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    metric_name: str,
    dataset_root: Path,
):
    font = load_font(11)

    thumbs = []

    for _, row in df.iterrows():
        img_path = get_image_path(row, dataset_root)

        try:
            img = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"Immagine saltata: {row.get('relative_path', '<no path>')} | errore: {e}")
            continue

        img.thumbnail(THUMB_SIZE)

        canvas = Image.new("L", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), color=255)

        x = (THUMB_SIZE[0] - img.width) // 2
        y = 0
        canvas.paste(img, (x, y))

        draw = ImageDraw.Draw(canvas)

        label_1 = str(row["relative_path"])
        metric_value = row[metric_name]

        if pd.isna(metric_value):
            label_2 = f"{metric_name}: NaN"
        else:
            label_2 = f"{metric_name}: {metric_value:.4f}"

        draw.text((4, THUMB_SIZE[1] + 3), label_1[:28], fill=0, font=font)
        draw.text((4, THUMB_SIZE[1] + 20), label_2, fill=0, font=font)

        thumbs.append(canvas)

    if not thumbs:
        print(f"Nessuna immagine per {title}")
        return

    grid_rows = (len(thumbs) + GRID_COLS - 1) // GRID_COLS

    sheet_width = GRID_COLS * THUMB_SIZE[0]
    sheet_height = grid_rows * (THUMB_SIZE[1] + LABEL_HEIGHT) + 40

    sheet = Image.new("L", (sheet_width, sheet_height), color=255)
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(16)
    draw.text((10, 10), title, fill=0, font=title_font)

    start_y = 40

    for idx, thumb in enumerate(thumbs):
        col = idx % GRID_COLS
        grid_row = idx // GRID_COLS

        x = col * THUMB_SIZE[0]
        y = start_y + grid_row * (THUMB_SIZE[1] + LABEL_HEIGHT)

        sheet.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"Salvato: {project_display_path(output_path)}")


def inspect_quality_outliers(
    quality_report: Path,
    dataset_root: Path,
    output_dir: Path,
    n_outliers: int,
):
    if not quality_report.exists():
        raise FileNotFoundError(f"Quality report non trovato: {quality_report}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_quality_report(quality_report)

    print("=== QUALITY OUTLIER INSPECTION ===")
    print(f"Quality report: {project_display_path(quality_report)}")
    print(f"Dataset root: {project_display_path(dataset_root)}")
    print(f"Immagini nel quality report: {len(df)}")
    print(f"Outlier per categoria: {n_outliers}")
    print()

    low_focus = df.sort_values("focus_score", ascending=True).head(n_outliers).copy()
    high_bright = df.sort_values("bright_ratio", ascending=False).head(n_outliers).copy()
    low_contrast = df.sort_values("contrast", ascending=True).head(n_outliers).copy()
    high_dark = df.sort_values("dark_ratio", ascending=False).head(n_outliers).copy()

    low_focus["outlier_type"] = "low_focus"
    high_bright["outlier_type"] = "high_bright_ratio"
    low_contrast["outlier_type"] = "low_contrast"
    high_dark["outlier_type"] = "high_dark_ratio"

    outliers = pd.concat(
        [low_focus, high_bright, low_contrast, high_dark],
        ignore_index=True,
    )

    outliers_csv = output_dir / "quality_outliers.csv"
    outliers.to_csv(outliers_csv, index=False)

    make_contact_sheet(
        low_focus,
        title="Lowest focus_score images",
        output_path=output_dir / "low_focus_contact_sheet.png",
        metric_name="focus_score",
        dataset_root=dataset_root,
    )

    make_contact_sheet(
        high_bright,
        title="Highest bright_ratio images",
        output_path=output_dir / "high_bright_ratio_contact_sheet.png",
        metric_name="bright_ratio",
        dataset_root=dataset_root,
    )

    make_contact_sheet(
        low_contrast,
        title="Lowest contrast images",
        output_path=output_dir / "low_contrast_contact_sheet.png",
        metric_name="contrast",
        dataset_root=dataset_root,
    )

    make_contact_sheet(
        high_dark,
        title="Highest dark_ratio images",
        output_path=output_dir / "high_dark_ratio_contact_sheet.png",
        metric_name="dark_ratio",
        dataset_root=dataset_root,
    )

    print()
    print("=== OUTLIER SUMMARY ===")
    print(f"Salvato CSV outlier in: {project_display_path(outliers_csv)}")
    print()

    print("Lowest focus_score:")
    print(
        low_focus[
            ["relative_path", "focus_score", "brightness", "contrast", "bright_ratio"]
        ].to_string(index=False)
    )

    print("\nHighest bright_ratio:")
    print(
        high_bright[
            ["relative_path", "bright_ratio", "brightness", "contrast", "focus_score"]
        ].to_string(index=False)
    )

    print("\nLowest contrast:")
    print(
        low_contrast[
            ["relative_path", "contrast", "brightness", "focus_score", "bright_ratio"]
        ].to_string(index=False)
    )

    print("\nHighest dark_ratio:")
    print(
        high_dark[
            ["relative_path", "dark_ratio", "brightness", "contrast", "focus_score"]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=DEFAULT_QUALITY_REPORT,
        help="Path a quality_report.csv. Default: data/quality/quality_report.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path alla cartella CASIA-Iris-Twins. Default: data/raw/CASIA-Iris-Twins",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella dove salvare gli outlier. Default: data/quality/outliers",
    )
    parser.add_argument(
        "--n-outliers",
        type=int,
        default=DEFAULT_N_OUTLIERS,
        help="Numero di outlier per categoria. Default: 25",
    )

    args = parser.parse_args()

    quality_report = resolve_from_project(args.quality_report)
    dataset_root = resolve_from_project(args.dataset_root)
    output_dir = resolve_from_project(args.output_dir)

    inspect_quality_outliers(
        quality_report=quality_report,
        dataset_root=dataset_root,
        output_dir=output_dir,
        n_outliers=args.n_outliers,
    )


if __name__ == "__main__":
    main()
