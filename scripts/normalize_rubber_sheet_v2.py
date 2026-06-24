from pathlib import Path
import json
import math

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_METADATA = PROJECT_ROOT / "data/metadata/metadata_segmented_strict.csv"

OUTPUT_DIR = PROJECT_ROOT / "data/normalized/strict_v2"
IMAGES_DIR = OUTPUT_DIR / "images"
MASKS_DIR = OUTPUT_DIR / "masks"
MASKED_PREVIEW_DIR = OUTPUT_DIR / "masked_previews"
CONTACT_DIR = OUTPUT_DIR / "contact_sheets"

REPORT_PATH = OUTPUT_DIR / "normalized_strict_v2_report.csv"
SUMMARY_PATH = OUTPUT_DIR / "normalization_v2_summary.json"

RADIAL_RES = 64
ANGULAR_RES = 512

THUMB_SIZE = (256, 64)
LABEL_HEIGHT = 38
CONTACT_COLS = 3


def safe_name(relative_path: str) -> str:
    return relative_path.replace("/", "__").replace("\\", "__")


def build_conservative_roi_mask(radial_res=64, angular_res=512):
    """
    Maschera geometrica conservativa nel dominio normalizzato.

    Righe = raggio, da bordo pupilla a bordo iride.
    Colonne = angolo theta.

    Manteniamo soprattutto le zone laterali dell'iride:
    - intorno a theta = 0
    - intorno a theta = pi

    Escludiamo:
    - zona molto vicina alla pupilla
    - zona molto vicina al limbus
    - settori superiore/inferiore, spesso coperti da palpebre/ciglia.
    """

    yy, xx = np.indices((radial_res, angular_res))

    theta = (xx / angular_res) * 2 * np.pi

    # Zone laterali: destra e sinistra.
    # Ampiezza: circa ±50 gradi intorno a 0 e 180.
    side_width = np.deg2rad(50)

    right_side = (theta <= side_width) | (theta >= 2 * np.pi - side_width)
    left_side = (theta >= np.pi - side_width) & (theta <= np.pi + side_width)

    angular_valid = right_side | left_side

    # Evitiamo i bordi radiali: vicino alla pupilla e vicino alla sclera.
    radial_valid = (yy >= 5) & (yy <= radial_res - 6)

    return angular_valid & radial_valid


def rubber_sheet_normalize_v2(gray, row, radial_res=64, angular_res=512):
    h, w = gray.shape

    pupil_x = float(row["pupil_x"])
    pupil_y = float(row["pupil_y"])
    pupil_r = float(row["pupil_r"])

    iris_x = float(row["iris_x"])
    iris_y = float(row["iris_y"])
    iris_r = float(row["iris_r"])

    theta = np.linspace(0, 2 * np.pi, angular_res, endpoint=False)

    # Bordo pupilla
    xp = pupil_x + pupil_r * np.cos(theta)
    yp = pupil_y + pupil_r * np.sin(theta)

    # Bordo iride
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

    # Maschera fotometrica.
    # Riflessi/specularità: pixel molto chiari.
    # Artefatti neri o bordi fuori immagine: pixel troppo scuri.
    too_bright = normalized >= 220
    too_dark = normalized <= 8

    bad_pixels = too_bright | too_dark

    # Dilata leggermente i pixel cattivi per rimuovere anche il bordo dei riflessi.
    bad_uint8 = bad_pixels.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bad_dilated = cv2.dilate(bad_uint8, kernel, iterations=1) > 0

    final_mask = inside_image & roi_mask & (~bad_dilated)

    mask_uint8 = final_mask.astype(np.uint8) * 255

    # Preview: mettiamo a grigio medio le zone non valide.
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
    if len(df) == 0:
        return

    if len(df) > n:
        df = df.sample(n=n, random_state=42).copy()

    font = ImageFont.load_default()
    thumbs = []

    for _, row in df.iterrows():
        path = Path(row[image_col])

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
    print(f"Salvata contact sheet: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    MASKED_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_METADATA,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    print("=== RUBBER SHEET NORMALIZATION V2 ===")
    print(f"Input metadata: {INPUT_METADATA}")
    print(f"Immagini da normalizzare: {len(df)}")
    print(f"Output size: {RADIAL_RES} x {ANGULAR_RES}")
    print()

    rows = []

    for idx, row in df.iterrows():
        image_path = Path(row["absolute_path"])
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        out_row = row.to_dict()

        if gray is None:
            out_row["normalization_ok"] = False
            out_row["normalization_error"] = "cv2.imread failed"
            rows.append(out_row)
            continue

        try:
            normalized, mask, preview, valid_total, valid_roi = rubber_sheet_normalize_v2(
                gray,
                row,
                radial_res=RADIAL_RES,
                angular_res=ANGULAR_RES,
            )

            base = safe_name(row["relative_path"])

            norm_path = IMAGES_DIR / f"{base}.png"
            mask_path = MASKS_DIR / f"{base}_mask.png"
            preview_path = MASKED_PREVIEW_DIR / f"{base}_preview.png"

            cv2.imwrite(str(norm_path), normalized)
            cv2.imwrite(str(mask_path), mask)
            cv2.imwrite(str(preview_path), preview)

            # Accettiamo se dentro la ROI laterale resta abbastanza area valida.
            normalization_ok = valid_roi >= 0.65

            out_row["normalization_ok"] = bool(normalization_ok)
            out_row["normalization_error"] = ""
            out_row["normalized_path"] = str(norm_path)
            out_row["mask_path"] = str(mask_path)
            out_row["masked_preview_path"] = str(preview_path)
            out_row["normalized_height"] = RADIAL_RES
            out_row["normalized_width"] = ANGULAR_RES
            out_row["valid_ratio_total"] = valid_total
            out_row["valid_ratio_roi"] = valid_roi

        except Exception as e:
            out_row["normalization_ok"] = False
            out_row["normalization_error"] = str(e)

        rows.append(out_row)

        if (idx + 1) % 250 == 0:
            print(f"Normalizzate {idx + 1}/{len(df)} immagini")

    ndf = pd.DataFrame(rows)
    ndf.to_csv(REPORT_PATH, index=False)

    ok_df = ndf[ndf["normalization_ok"] == True].copy()

    summary = {
        "input_images": int(len(df)),
        "normalization_ok": int(len(ok_df)),
        "normalization_failed_or_rejected": int(len(df) - len(ok_df)),
        "radial_res": RADIAL_RES,
        "angular_res": ANGULAR_RES,
    }

    if len(ndf) > 0:
        for col in ["valid_ratio_total", "valid_ratio_roi"]:
            values = pd.to_numeric(ndf[col], errors="coerce").dropna()
            summary[col] = {
                "min": float(values.min()),
                "q1": float(values.quantile(0.25)),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "q3": float(values.quantile(0.75)),
                "max": float(values.max()),
            }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_contact_sheet(
        ok_df,
        image_col="normalized_path",
        title="Rubber Sheet V2 normalized examples",
        output_path=CONTACT_DIR / "normalized_v2_examples.png",
        n=45,
    )

    make_contact_sheet(
        ok_df,
        image_col="masked_preview_path",
        title="Rubber Sheet V2 masked preview examples",
        output_path=CONTACT_DIR / "masked_preview_v2_examples.png",
        n=45,
    )

    make_contact_sheet(
        ok_df,
        image_col="mask_path",
        title="Rubber Sheet V2 mask examples",
        output_path=CONTACT_DIR / "mask_v2_examples.png",
        n=45,
    )

    low_valid = ndf.sort_values("valid_ratio_roi", ascending=True).head(45)
    make_contact_sheet(
        low_valid,
        image_col="masked_preview_path",
        title="Rubber Sheet V2 lowest ROI valid examples",
        output_path=CONTACT_DIR / "lowest_valid_roi_v2_examples.png",
        n=45,
    )

    print()
    print("=== NORMALIZATION V2 SUMMARY ===")
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

    print(f"Report salvato in:  {REPORT_PATH}")
    print(f"Summary salvato in: {SUMMARY_PATH}")
    print(f"Immagini in:        {IMAGES_DIR}")
    print(f"Maschere in:        {MASKS_DIR}")
    print(f"Preview in:         {MASKED_PREVIEW_DIR}")
    print(f"Contact sheet in:   {CONTACT_DIR}")


if __name__ == "__main__":
    main()