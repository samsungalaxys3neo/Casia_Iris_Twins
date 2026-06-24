from pathlib import Path
import math
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEGMENTATION_REPORT = PROJECT_ROOT / "data/segmentation_v2/segmentation_v2_report.csv"

OUTPUT_STRICT = PROJECT_ROOT / "data/metadata/metadata_segmented_strict.csv"
OUTPUT_REJECTED = PROJECT_ROOT / "data/metadata/metadata_segmented_strict_rejected.csv"

CONTACT_DIR = PROJECT_ROOT / "data/segmentation_v2/strict_contact_sheets"

THUMB_SIZE = (180, 135)
LABEL_HEIGHT = 42
COLS = 5


def make_contact_sheet(df, title, output_path, n=50):
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()

    if len(df) > n:
        df = df.sample(n=n, random_state=42).copy()

    thumbs = []

    for _, row in df.iterrows():
        overlay_name = row["relative_path"].replace("/", "__").replace("\\", "__") + ".png"
        overlay_path = PROJECT_ROOT / "data/segmentation_v2/overlays" / overlay_name

        if not overlay_path.exists():
            continue

        img = Image.open(overlay_path).convert("RGB")
        img.thumbnail(THUMB_SIZE)

        canvas = Image.new("RGB", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), "white")
        x = (THUMB_SIZE[0] - img.width) // 2
        canvas.paste(img, (x, 0))

        draw = ImageDraw.Draw(canvas)

        ratio = row.get("iris_pupil_radius_ratio", "")
        offset = row.get("iris_center_offset", "")
        score = row.get("iris_score", "")

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
        "white"
    )

    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black", font=font)

    y0 = 35

    for i, thumb in enumerate(thumbs):
        x = (i % COLS) * THUMB_SIZE[0]
        y = y0 + (i // COLS) * (THUMB_SIZE[1] + LABEL_HEIGHT)
        sheet.paste(thumb, (x, y))

    sheet.save(output_path)
    print(f"Salvata contact sheet: {output_path}")


def main():
    df = pd.read_csv(
        SEGMENTATION_REPORT,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    df["iris_confidence"] = df["iris_confidence"].fillna("none")

    # Filtro più severo.
    # Non è "verità assoluta", è un filtro conservativo per ridurre segmentazioni brutte.
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

    OUTPUT_STRICT.parent.mkdir(parents=True, exist_ok=True)

    df_strict.to_csv(OUTPUT_STRICT, index=False)
    df_rejected.to_csv(OUTPUT_REJECTED, index=False)

    print("=== STRICT SEGMENTATION FILTER ===")
    print(f"Input totale:              {len(df)}")
    print(f"Accettate strict:          {len(df_strict)}")
    print(f"Rifiutate strict:          {len(df_rejected)}")
    print(f"Percentuale accettata:     {len(df_strict) / len(df):.4f}")
    print()

    print("=== Distribuzione per folder_class strict ===")
    df_strict["folder_class"] = df_strict["twin_id"] + df_strict["eye"]
    print(df_strict["folder_class"].value_counts().sort_index().to_string())
    print()

    print("=== Famiglie rappresentate ===")
    print(f"Famiglie strict: {df_strict['family_id'].nunique()}")
    print()

    print("=== Statistiche strict ===")
    for col in ["pupil_r", "iris_r", "iris_pupil_radius_ratio", "iris_center_offset", "iris_score"]:
        print(col)
        print(df_strict[col].describe().to_string())
        print()

    make_contact_sheet(
        df_strict,
        "STRICT accepted segmentation examples",
        CONTACT_DIR / "strict_accepted_examples.png",
        n=50
    )

    make_contact_sheet(
        df_rejected[df_rejected["segmentation_ok"] == True],
        "Rejected despite segmentation_ok examples",
        CONTACT_DIR / "strict_rejected_but_ok_examples.png",
        n=50
    )

    print(f"Salvato strict metadata in: {OUTPUT_STRICT}")
    print(f"Salvato rejected metadata in: {OUTPUT_REJECTED}")


if __name__ == "__main__":
    main()