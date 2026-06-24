from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = PROJECT_ROOT / "data/segmentation_daugman/segmentation_daugman_report.csv"

OUTPUT_HIGH = PROJECT_ROOT / "data/metadata/metadata_daugman_high.csv"
OUTPUT_STRICT = PROJECT_ROOT / "data/metadata/metadata_daugman_strict.csv"
OUTPUT_REJECTED = PROJECT_ROOT / "data/metadata/metadata_daugman_rejected.csv"

CONTACT_DIR = PROJECT_ROOT / "data/segmentation_daugman/strict_contact_sheets"

THUMB_SIZE = (180, 135)
LABEL_HEIGHT = 42
COLS = 5


def make_contact_sheet(df, title, output_path, n=50):
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print(f"Nessuna immagine per {title}")
        return

    if len(df) > n:
        df = df.sample(n=n, random_state=42).copy()

    font = ImageFont.load_default()
    thumbs = []

    for _, row in df.iterrows():
        overlay_name = row["relative_path"].replace("/", "__").replace("\\", "__") + ".png"
        overlay_path = PROJECT_ROOT / "data/segmentation_daugman/overlays" / overlay_name

        if not overlay_path.exists():
            continue

        img = Image.open(overlay_path).convert("RGB")
        img.thumbnail(THUMB_SIZE)

        canvas = Image.new("RGB", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), "white")
        x = (THUMB_SIZE[0] - img.width) // 2
        canvas.paste(img, (x, 0))

        draw = ImageDraw.Draw(canvas)

        ratio = float(row["iris_pupil_radius_ratio"]) if pd.notna(row["iris_pupil_radius_ratio"]) else -1
        offset = float(row["iris_center_offset"]) if pd.notna(row["iris_center_offset"]) else -1
        score = float(row["iris_score"]) if pd.notna(row["iris_score"]) else -1

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
    print(f"Salvata contact sheet: {output_path}")


def main():
    df = pd.read_csv(
        INPUT,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    df["iris_confidence"] = df["iris_confidence"].fillna("none")

    # Primo subset: tutti gli high confidence accettati
    high_mask = (
        (df["segmentation_ok"] == True)
        & (df["iris_confidence"] == "high")
    )

    df_high = df[high_mask].copy()

    # Filtro strict: più conservativo.
    # Serve a tenere solo segmentazioni geometricamente più stabili.
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

    OUTPUT_HIGH.parent.mkdir(parents=True, exist_ok=True)

    df_high.to_csv(OUTPUT_HIGH, index=False)
    df_strict.to_csv(OUTPUT_STRICT, index=False)
    df_rejected.to_csv(OUTPUT_REJECTED, index=False)

    print("=== DAUGMAN STRICT FILTER ===")
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
        CONTACT_DIR / "daugman_strict_accepted_examples.png",
        n=50,
    )

    make_contact_sheet(
        df_rejected[df_rejected["segmentation_ok"] == True],
        "Daugman rejected despite segmentation_ok examples",
        CONTACT_DIR / "daugman_strict_rejected_but_ok_examples.png",
        n=50,
    )

    print(f"Salvato HIGH in:     {OUTPUT_HIGH}")
    print(f"Salvato STRICT in:   {OUTPUT_STRICT}")
    print(f"Salvato REJECTED in: {OUTPUT_REJECTED}")


if __name__ == "__main__":
    main()