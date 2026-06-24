from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


QUALITY_REPORT = Path("iris_twins_project/data/quality/quality_report.csv")
OUTPUT_DIR = Path("iris_twins_project/data/quality/outliers")

N_OUTLIERS = 25
THUMB_SIZE = (160, 120)
LABEL_HEIGHT = 42
GRID_COLS = 5


def load_font(size=12):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_contact_sheet(df, title, output_path, metric_name):
    font = load_font(11)

    rows = []
    for _, row in df.iterrows():
        img_path = Path(row["absolute_path"])

        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            continue

        img.thumbnail(THUMB_SIZE)

        canvas = Image.new("L", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), color=255)

        x = (THUMB_SIZE[0] - img.width) // 2
        y = 0
        canvas.paste(img, (x, y))

        draw = ImageDraw.Draw(canvas)
        label_1 = row["relative_path"]
        label_2 = f"{metric_name}: {row[metric_name]:.4f}"

        draw.text((4, THUMB_SIZE[1] + 3), label_1[:28], fill=0, font=font)
        draw.text((4, THUMB_SIZE[1] + 20), label_2, fill=0, font=font)

        rows.append(canvas)

    if not rows:
        print(f"Nessuna immagine per {title}")
        return

    grid_rows = (len(rows) + GRID_COLS - 1) // GRID_COLS

    sheet_width = GRID_COLS * THUMB_SIZE[0]
    sheet_height = grid_rows * (THUMB_SIZE[1] + LABEL_HEIGHT) + 40

    sheet = Image.new("L", (sheet_width, sheet_height), color=255)
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(16)
    draw.text((10, 10), title, fill=0, font=title_font)

    start_y = 40

    for idx, thumb in enumerate(rows):
        col = idx % GRID_COLS
        grid_row = idx // GRID_COLS

        x = col * THUMB_SIZE[0]
        y = start_y + grid_row * (THUMB_SIZE[1] + LABEL_HEIGHT)

        sheet.paste(thumb, (x, y))

    sheet.save(output_path)
    print(f"Salvato: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        QUALITY_REPORT,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    print("=== QUALITY OUTLIER INSPECTION ===")
    print(f"Immagini nel quality report: {len(df)}")

    low_focus = df.sort_values("focus_score", ascending=True).head(N_OUTLIERS).copy()
    high_bright = df.sort_values("bright_ratio", ascending=False).head(N_OUTLIERS).copy()
    low_contrast = df.sort_values("contrast", ascending=True).head(N_OUTLIERS).copy()
    high_dark = df.sort_values("dark_ratio", ascending=False).head(N_OUTLIERS).copy()

    low_focus["outlier_type"] = "low_focus"
    high_bright["outlier_type"] = "high_bright_ratio"
    low_contrast["outlier_type"] = "low_contrast"
    high_dark["outlier_type"] = "high_dark_ratio"

    outliers = pd.concat(
        [low_focus, high_bright, low_contrast, high_dark],
        ignore_index=True
    )

    outliers_csv = OUTPUT_DIR / "quality_outliers.csv"
    outliers.to_csv(outliers_csv, index=False)

    make_contact_sheet(
        low_focus,
        title="Lowest focus_score images",
        output_path=OUTPUT_DIR / "low_focus_contact_sheet.png",
        metric_name="focus_score",
    )

    make_contact_sheet(
        high_bright,
        title="Highest bright_ratio images",
        output_path=OUTPUT_DIR / "high_bright_ratio_contact_sheet.png",
        metric_name="bright_ratio",
    )

    make_contact_sheet(
        low_contrast,
        title="Lowest contrast images",
        output_path=OUTPUT_DIR / "low_contrast_contact_sheet.png",
        metric_name="contrast",
    )

    make_contact_sheet(
        high_dark,
        title="Highest dark_ratio images",
        output_path=OUTPUT_DIR / "high_dark_ratio_contact_sheet.png",
        metric_name="dark_ratio",
    )

    print()
    print("=== OUTLIER SUMMARY ===")
    print(f"Salvato CSV outlier in: {outliers_csv}")
    print()

    print("Lowest focus_score:")
    print(low_focus[["relative_path", "focus_score", "brightness", "contrast", "bright_ratio"]].to_string(index=False))

    print("\nHighest bright_ratio:")
    print(high_bright[["relative_path", "bright_ratio", "brightness", "contrast", "focus_score"]].to_string(index=False))

    print("\nLowest contrast:")
    print(low_contrast[["relative_path", "contrast", "brightness", "focus_score", "bright_ratio"]].to_string(index=False))

    print("\nHighest dark_ratio:")
    print(high_dark[["relative_path", "dark_ratio", "brightness", "contrast", "focus_score"]].to_string(index=False))


if __name__ == "__main__":
    main()