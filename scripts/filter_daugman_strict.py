from pathlib import Path
import argparse
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "segmentation_daugman" / "segmentation_daugman_report.csv"

DEFAULT_OUTPUT_HIGH = PROJECT_ROOT / "data" / "metadata" / "metadata_daugman_high.csv"
DEFAULT_OUTPUT_STRICT = PROJECT_ROOT / "data" / "metadata" / "metadata_daugman_strict.csv"
DEFAULT_OUTPUT_REJECTED = PROJECT_ROOT / "data" / "metadata" / "metadata_daugman_rejected.csv"

DEFAULT_OVERLAY_DIR = PROJECT_ROOT / "data" / "segmentation_daugman" / "overlays"
DEFAULT_CONTACT_DIR = PROJECT_ROOT / "data" / "segmentation_daugman" / "strict_contact_sheets"

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


def read_segmentation_report(input_path: Path) -> pd.DataFrame:
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

    df = pd.read_csv(input_path, dtype=dtype_map)

    # Rende robusta la colonna booleana anche se letta come stringa.
    df["segmentation_ok"] = (
        df["segmentation_ok"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    df["iris_confidence"] = df["iris_confidence"].fillna("none").astype(str)

    numeric_cols = [
        "pupil_r",
        "iris_r",
        "iris_pupil_radius_ratio",
        "iris_center_offset",
        "iris_score",
        "pupil_inner_mean",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def overlay_name_from_relative_path(relative_path: str) -> str:
    return str(relative_path).replace("/", "__").replace("\\", "__") + ".png"


def make_contact_sheet(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    overlay_dir: Path,
    n: int = 50,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print(f"Nessuna immagine per {title}")
        return

    if len(df) > n:
        df = df.sample(n=n, random_state=42).copy()

    font = ImageFont.load_default()
    thumbs = []

    for _, row in df.iterrows():
        overlay_name = overlay_name_from_relative_path(row["relative_path"])
        overlay_path = overlay_dir / overlay_name

        if not overlay_path.exists():
            continue

        try:
            img = Image.open(overlay_path).convert("RGB")
        except Exception:
            continue

        img.thumbnail(THUMB_SIZE)

        canvas = Image.new("RGB", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), "white")
        x = (THUMB_SIZE[0] - img.width) // 2
        canvas.paste(img, (x, 0))

        draw = ImageDraw.Draw(canvas)

        ratio = row["iris_pupil_radius_ratio"] if pd.notna(row["iris_pupil_radius_ratio"]) else -1
        offset = row["iris_center_offset"] if pd.notna(row["iris_center_offset"]) else -1
        score = row["iris_score"] if pd.notna(row["iris_score"]) else -1

        label1 = str(row["relative_path"])[:32]
        label2 = f"ratio={ratio:.2f} off={offset:.1f} score={score:.1f}"

        draw.text((4, THUMB_SIZE[1] + 3), label1, fill="black", font=font)
        draw.text((4, THUMB_SIZE[1] + 20), label2, fill="black", font=font)

        thumbs.append(canvas)

    if not thumbs:
        print(f"Nessun overlay trovato per {title}")
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


def run_filter(
    input_path: Path,
    output_high: Path,
    output_strict: Path,
    output_rejected: Path,
    overlay_dir: Path,
    contact_dir: Path,
):
    if not input_path.exists():
        raise FileNotFoundError(f"Report Daugman non trovato: {input_path}")

    if not overlay_dir.exists():
        print(f"Attenzione: overlay directory non trovata: {overlay_dir}")
        print("Il filtro CSV verrà creato comunque, ma le contact sheet potrebbero essere vuote.")

    df = read_segmentation_report(input_path)

    high_mask = (
        (df["segmentation_ok"] == True)
        & (df["iris_confidence"] == "high")
    )

    df_high = df[high_mask].copy()

    strict_mask = (
        (df["segmentation_ok"] == True)
        & (df["iris_confidence"] == "high")
        & (df["pupil_r"] >= 25)
        & (df["pupil_r"] <= 80)
        & (df["iris_r"] >= 90)
        & (df["iris_r"] <= 155)
        & (df["iris_pupil_radius_ratio"] >= 2.0)
        & (df["iris_pupil_radius_ratio"] <= 3.1)
        & (df["iris_center_offset"] <= 18)
        & (df["iris_score"] >= 45)
        & (df["pupil_inner_mean"] <= 95)
    )

    df_strict = df[strict_mask].copy()
    df_rejected = df[~strict_mask].copy()

    output_high.parent.mkdir(parents=True, exist_ok=True)

    df_high.to_csv(output_high, index=False)
    df_strict.to_csv(output_strict, index=False)
    df_rejected.to_csv(output_rejected, index=False)

    print("=== DAUGMAN STRICT FILTER ===")
    print(f"Input:                    {project_display_path(input_path)}")
    print(f"Overlay dir:              {project_display_path(overlay_dir)}")
    print(f"Input totale:             {len(df)}")
    print(f"High confidence:          {len(df_high)}")
    print(f"Strict accepted:          {len(df_strict)}")
    print(f"Rejected:                 {len(df_rejected)}")
    print(f"Strict acceptance rate:   {len(df_strict) / len(df):.4f}")
    print()

    print("=== Distribuzione confidence originale ===")
    print(df["iris_confidence"].value_counts(dropna=False).to_string())
    print()

    print("=== Distribuzione folder_class strict ===")
    df_strict["folder_class"] = df_strict["twin_id"] + df_strict["eye"]
    print(df_strict["folder_class"].value_counts().sort_index().to_string())
    print()

    print("=== Famiglie rappresentate strict ===")
    print(df_strict["family_id"].nunique())
    print()

    print("=== Statistiche strict ===")
    for col in [
        "pupil_r",
        "iris_r",
        "iris_pupil_radius_ratio",
        "iris_center_offset",
        "iris_score",
        "pupil_inner_mean",
    ]:
        print(col)
        print(df_strict[col].describe().to_string())
        print()

    make_contact_sheet(
        df_strict,
        "Daugman STRICT accepted examples",
        contact_dir / "daugman_strict_accepted_examples.png",
        overlay_dir=overlay_dir,
        n=50,
    )

    make_contact_sheet(
        df_rejected[df_rejected["segmentation_ok"] == True],
        "Daugman rejected despite segmentation_ok examples",
        contact_dir / "daugman_strict_rejected_but_ok_examples.png",
        overlay_dir=overlay_dir,
        n=50,
    )

    print(f"Salvato HIGH in:     {project_display_path(output_high)}")
    print(f"Salvato STRICT in:   {project_display_path(output_strict)}")
    print(f"Salvato REJECTED in: {project_display_path(output_rejected)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path al segmentation_daugman_report.csv. Default: data/segmentation_daugman/segmentation_daugman_report.csv",
    )
    parser.add_argument(
        "--output-high",
        type=Path,
        default=DEFAULT_OUTPUT_HIGH,
        help="Path output metadata_daugman_high.csv.",
    )
    parser.add_argument(
        "--output-strict",
        type=Path,
        default=DEFAULT_OUTPUT_STRICT,
        help="Path output metadata_daugman_strict.csv.",
    )
    parser.add_argument(
        "--output-rejected",
        type=Path,
        default=DEFAULT_OUTPUT_REJECTED,
        help="Path output metadata_daugman_rejected.csv.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=DEFAULT_OVERLAY_DIR,
        help="Cartella overlay Daugman. Default: data/segmentation_daugman/overlays",
    )
    parser.add_argument(
        "--contact-dir",
        type=Path,
        default=DEFAULT_CONTACT_DIR,
        help="Cartella contact sheets strict. Default: data/segmentation_daugman/strict_contact_sheets",
    )

    args = parser.parse_args()

    run_filter(
        input_path=resolve_from_project(args.input),
        output_high=resolve_from_project(args.output_high),
        output_strict=resolve_from_project(args.output_strict),
        output_rejected=resolve_from_project(args.output_rejected),
        overlay_dir=resolve_from_project(args.overlay_dir),
        contact_dir=resolve_from_project(args.contact_dir),
    )


if __name__ == "__main__":
    main()
