from pathlib import Path
import argparse
import json
import math

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_METADATA = PROJECT_ROOT / "data" / "metadata" / "metadata_daugman_strict.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "CASIA-Iris-Twins"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "normalized" / "daugman_strict"

RADIAL_RES = 64
ANGULAR_RES = 512

THUMB_SIZE = (256, 64)
LABEL_HEIGHT = 38
CONTACT_COLS = 3

# Maschera conservativa: usiamo soprattutto le regioni laterali dell'iride.
SIDE_WIDTH_DEG = 55

MIN_ROI_VALID_RATIO = 0.65


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


def project_relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_metadata(metadata_path: Path) -> pd.DataFrame:
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

    df = pd.read_csv(metadata_path, dtype=dtype_map)

    numeric_cols = [
        "pupil_x",
        "pupil_y",
        "pupil_r",
        "iris_x",
        "iris_y",
        "iris_r",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_image_path(row: pd.Series, dataset_root: Path) -> Path:
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


def safe_name(relative_path: str) -> str:
    return str(relative_path).replace("/", "__").replace("\\", "__")


def build_conservative_roi_mask(radial_res=64, angular_res=512):
    yy, xx = np.indices((radial_res, angular_res))
    theta = (xx / angular_res) * 2 * np.pi

    side_width = np.deg2rad(SIDE_WIDTH_DEG)

    right_side = (theta <= side_width) | (theta >= 2 * np.pi - side_width)
    left_side = (theta >= np.pi - side_width) & (theta <= np.pi + side_width)

    angular_valid = right_side | left_side

    # Evitiamo zona troppo vicina alla pupilla e al bordo esterno dell'iride.
    radial_valid = (yy >= 5) & (yy <= radial_res - 6)

    return angular_valid & radial_valid


def rubber_sheet_normalize(gray, row, radial_res=64, angular_res=512):
    h, w = gray.shape

    pupil_x = float(row["pupil_x"])
    pupil_y = float(row["pupil_y"])
    pupil_r = float(row["pupil_r"])

    iris_x = float(row["iris_x"])
    iris_y = float(row["iris_y"])
    iris_r = float(row["iris_r"])

    theta = np.linspace(0, 2 * np.pi, angular_res, endpoint=False)

    xp = pupil_x + pupil_r * np.cos(theta)
    yp = pupil_y + pupil_r * np.sin(theta)

    xi = iris_x + iris_r * np.cos(theta)
    yi = iris_y + iris_r * np.sin(theta)

    r = np.linspace(0, 1, radial_res).reshape(-1, 1)

    map_x = (1 - r) * xp.reshape(1, -1) + r * xi.reshape(1, -1)
    map_y = (1 - r) * yp.reshape(1, -1) + r * yi.reshape(1, -1)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    normalized = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    inside_image = (
        (map_x >= 0)
        & (map_x < w)
        & (map_y >= 0)
        & (map_y < h)
    )

    roi_mask = build_conservative_roi_mask(radial_res, angular_res)

    too_bright = normalized >= 220
    too_dark = normalized <= 8

    bad_pixels = too_bright | too_dark

    bad_uint8 = bad_pixels.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bad_dilated = cv2.dilate(bad_uint8, kernel, iterations=1) > 0

    final_mask = inside_image & roi_mask & (~bad_dilated)

    mask_uint8 = final_mask.astype(np.uint8) * 255

    preview = normalized.copy()
    preview[~final_mask] = 128

    total_valid_ratio = float(np.mean(final_mask))

    roi_area = np.sum(inside_image & roi_mask)
    if roi_area > 0:
        roi_valid_ratio = float(np.sum(final_mask) / roi_area)
    else:
        roi_valid_ratio = 0.0

    return normalized, mask_uint8, preview, total_valid_ratio, roi_valid_ratio


def make_contact_sheet(df, image_col, title, output_path, n=45):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        return

    if len(df) > n:
        df = df.sample(n=n, random_state=42).copy()

    font = ImageFont.load_default()
    thumbs = []

    for _, row in df.iterrows():
        path_value = row.get(image_col, None)

        if pd.isna(path_value):
            continue

        path = resolve_from_project(Path(str(path_value)))

        if not path.exists():
            continue

        img = Image.open(path).convert("L")
        img = img.resize(THUMB_SIZE)

        canvas = Image.new("L", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), color=255)
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        label1 = str(row["relative_path"])[:42]
        label2 = f"total={row['valid_ratio_total']:.3f} roi={row['valid_ratio_roi']:.3f}"

        draw.text((4, THUMB_SIZE[1] + 3), label1, fill=0, font=font)
        draw.text((4, THUMB_SIZE[1] + 19), label2, fill=0, font=font)

        thumbs.append(canvas)

    if not thumbs:
        return

    rows = math.ceil(len(thumbs) / CONTACT_COLS)

    sheet = Image.new(
        "L",
        (CONTACT_COLS * THUMB_SIZE[0], rows * (THUMB_SIZE[1] + LABEL_HEIGHT) + 35),
        color=255,
    )

    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=0, font=font)

    y0 = 35

    for i, thumb in enumerate(thumbs):
        x = (i % CONTACT_COLS) * THUMB_SIZE[0]
        y = y0 + (i // CONTACT_COLS) * (THUMB_SIZE[1] + LABEL_HEIGHT)
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)
    print(f"Salvata contact sheet: {project_display_path(output_path)}")


def summarize_values(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return {
            "min": None,
            "q1": None,
            "median": None,
            "mean": None,
            "q3": None,
            "max": None,
        }

    return {
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def run_normalization(
    input_metadata: Path,
    dataset_root: Path,
    output_dir: Path,
):
    if not input_metadata.exists():
        raise FileNotFoundError(f"Input metadata non trovato: {input_metadata}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    masked_preview_dir = output_dir / "masked_previews"
    contact_dir = output_dir / "contact_sheets"

    report_path = output_dir / "normalized_daugman_strict_report.csv"
    summary_path = output_dir / "normalization_daugman_strict_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    masked_preview_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    df = read_metadata(input_metadata)

    print("=== DAUGMAN STRICT RUBBER SHEET NORMALIZATION ===")
    print(f"Input metadata: {project_display_path(input_metadata)}")
    print(f"Dataset root: {project_display_path(dataset_root)}")
    print(f"Output dir: {project_display_path(output_dir)}")
    print(f"Immagini da normalizzare: {len(df)}")
    print(f"Output size: {RADIAL_RES} x {ANGULAR_RES}")
    print(f"Side ROI width: ±{SIDE_WIDTH_DEG} degrees")
    print()

    rows = []

    for processed_idx, (_, row) in enumerate(df.iterrows(), start=1):
        out_row = row.to_dict()

        try:
            image_path = get_image_path(row, dataset_root)
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if gray is None:
                out_row["normalization_ok"] = False
                out_row["normalization_error"] = f"cv2.imread failed: {image_path}"
                rows.append(out_row)
                continue

            normalized, mask, preview, valid_total, valid_roi = rubber_sheet_normalize(
                gray,
                row,
                radial_res=RADIAL_RES,
                angular_res=ANGULAR_RES,
            )

            base = safe_name(row["relative_path"])

            norm_path = images_dir / f"{base}.png"
            mask_path = masks_dir / f"{base}_mask.png"
            preview_path = masked_preview_dir / f"{base}_preview.png"

            cv2.imwrite(str(norm_path), normalized)
            cv2.imwrite(str(mask_path), mask)
            cv2.imwrite(str(preview_path), preview)

            normalization_ok = valid_roi >= MIN_ROI_VALID_RATIO

            out_row["normalization_ok"] = bool(normalization_ok)
            out_row["normalization_error"] = ""
            out_row["normalized_path"] = project_relative_or_absolute(norm_path)
            out_row["mask_path"] = project_relative_or_absolute(mask_path)
            out_row["masked_preview_path"] = project_relative_or_absolute(preview_path)
            out_row["normalized_height"] = RADIAL_RES
            out_row["normalized_width"] = ANGULAR_RES
            out_row["valid_ratio_total"] = valid_total
            out_row["valid_ratio_roi"] = valid_roi

        except Exception as e:
            out_row["normalization_ok"] = False
            out_row["normalization_error"] = str(e)

        rows.append(out_row)

        if processed_idx % 250 == 0:
            print(f"Normalizzate {processed_idx}/{len(df)} immagini")

    ndf = pd.DataFrame(rows)
    ndf.to_csv(report_path, index=False)

    ok_df = ndf[ndf["normalization_ok"] == True].copy()

    summary = {
        "input_images": int(len(df)),
        "normalization_ok": int(len(ok_df)),
        "normalization_failed_or_rejected": int(len(df) - len(ok_df)),
        "input_metadata": project_display_path(input_metadata),
        "dataset_root": project_display_path(dataset_root),
        "output_dir": project_display_path(output_dir),
        "radial_res": RADIAL_RES,
        "angular_res": ANGULAR_RES,
        "side_width_deg": SIDE_WIDTH_DEG,
        "min_roi_valid_ratio": MIN_ROI_VALID_RATIO,
    }

    for col in ["valid_ratio_total", "valid_ratio_roi"]:
        if col in ndf.columns:
            summary[col] = summarize_values(ndf[col])

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_contact_sheet(
        ok_df,
        image_col="normalized_path",
        title="Daugman strict Rubber Sheet normalized examples",
        output_path=contact_dir / "normalized_examples.png",
        n=45,
    )

    make_contact_sheet(
        ok_df,
        image_col="masked_preview_path",
        title="Daugman strict masked preview examples",
        output_path=contact_dir / "masked_preview_examples.png",
        n=45,
    )

    make_contact_sheet(
        ok_df,
        image_col="mask_path",
        title="Daugman strict mask examples",
        output_path=contact_dir / "mask_examples.png",
        n=45,
    )

    if "valid_ratio_roi" in ndf.columns:
        low_valid = ndf.sort_values("valid_ratio_roi", ascending=True).head(45)
        make_contact_sheet(
            low_valid,
            image_col="masked_preview_path",
            title="Daugman strict lowest ROI valid examples",
            output_path=contact_dir / "lowest_valid_roi_examples.png",
            n=45,
        )

    print()
    print("=== NORMALIZATION SUMMARY ===")
    print(f"Input immagini:                  {summary['input_images']}")
    print(f"Normalization OK:                {summary['normalization_ok']}")
    print(f"Failed/rejected:                 {summary['normalization_failed_or_rejected']}")

    print()
    for col in ["valid_ratio_total", "valid_ratio_roi"]:
        if col in summary:
            print(col)
            for k, v in summary[col].items():
                print(f"  {k}: {v}")
            print()

    print(f"Report salvato in:  {project_display_path(report_path)}")
    print(f"Summary salvato in: {project_display_path(summary_path)}")
    print(f"Immagini in:        {project_display_path(images_dir)}")
    print(f"Maschere in:        {project_display_path(masks_dir)}")
    print(f"Preview in:         {project_display_path(masked_preview_dir)}")
    print(f"Contact sheet in:   {project_display_path(contact_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_INPUT_METADATA,
        help="Path a metadata_daugman_strict.csv. Default: data/metadata/metadata_daugman_strict.csv",
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
        help="Cartella output. Default: data/normalized/daugman_strict",
    )

    args = parser.parse_args()

    run_normalization(
        input_metadata=resolve_from_project(args.metadata),
        dataset_root=resolve_from_project(args.dataset_root),
        output_dir=resolve_from_project(args.output_dir),
    )


if __name__ == "__main__":
    main()
