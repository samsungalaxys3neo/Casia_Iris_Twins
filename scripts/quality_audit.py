from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METADATA_PATH = Path("iris_twins_project/data/metadata/metadata_clean.csv")
OUTPUT_DIR = Path("iris_twins_project/data/quality")
QUALITY_CSV = OUTPUT_DIR / "quality_report.csv"
SUMMARY_JSON = OUTPUT_DIR / "quality_summary.json"
PLOTS_DIR = OUTPUT_DIR / "plots"


def compute_quality(image_path: Path) -> dict:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {
            "quality_ok": False,
            "brightness": None,
            "contrast": None,
            "focus_score": None,
            "dark_ratio": None,
            "bright_ratio": None,
            "error": "cv2.imread failed",
        }

    img_float = img.astype(np.float32)

    brightness = float(np.mean(img_float))
    contrast = float(np.std(img_float))

    # Varianza del Laplaciano: più è alta, più l'immagine tende a essere nitida
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    focus_score = float(laplacian.var())

    dark_ratio = float(np.mean(img < 20))
    bright_ratio = float(np.mean(img > 235))

    return {
        "quality_ok": True,
        "brightness": brightness,
        "contrast": contrast,
        "focus_score": focus_score,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "error": "",
    }


def save_histogram(df: pd.DataFrame, column: str, output_path: Path, bins: int = 50):
    plt.figure()
    plt.hist(df[column].dropna(), bins=bins)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_boxplot_by_eye(df: pd.DataFrame, column: str, output_path: Path):
    groups = []
    labels = []

    for label in ["1L", "1R", "2L", "2R"]:
        sub = df[df["folder_class"] == label][column].dropna()
        if len(sub) > 0:
            groups.append(sub)
            labels.append(label)

    plt.figure()
    plt.boxplot(groups, labels=labels)
    plt.title(f"{column} by folder class")
    plt.xlabel("folder class")
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        METADATA_PATH,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    rows = []

    print("=== QUALITY AUDIT ===")
    print(f"Immagini da processare: {len(df)}")

    for idx, row in df.iterrows():
        image_path = Path(row["absolute_path"])
        quality = compute_quality(image_path)

        output_row = row.to_dict()
        output_row.update(quality)
        output_row["folder_class"] = f"{row['twin_id']}{row['eye']}"

        rows.append(output_row)

        if (idx + 1) % 250 == 0:
            print(f"Processate {idx + 1}/{len(df)} immagini")

    qdf = pd.DataFrame(rows)
    qdf.to_csv(QUALITY_CSV, index=False)

    numeric_cols = [
        "brightness",
        "contrast",
        "focus_score",
        "dark_ratio",
        "bright_ratio",
    ]

    summary = {
        "n_images": int(len(qdf)),
        "n_quality_ok": int(qdf["quality_ok"].sum()),
        "n_quality_failed": int((~qdf["quality_ok"]).sum()),
        "metrics": {},
    }

    for col in numeric_cols:
        values = qdf[col].dropna()
        summary["metrics"][col] = {
            "min": float(values.min()),
            "q1": float(values.quantile(0.25)),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "q3": float(values.quantile(0.75)),
            "max": float(values.max()),
        }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for col in numeric_cols:
        save_histogram(qdf, col, PLOTS_DIR / f"hist_{col}.png")
        save_boxplot_by_eye(qdf, col, PLOTS_DIR / f"box_{col}_by_folder_class.png")

    print()
    print("=== QUALITY SUMMARY ===")
    print(f"Quality ok:     {summary['n_quality_ok']}")
    print(f"Quality failed: {summary['n_quality_failed']}")
    print()

    for col, stats in summary["metrics"].items():
        print(f"{col}")
        print(f"  min:    {stats['min']:.4f}")
        print(f"  q1:     {stats['q1']:.4f}")
        print(f"  median: {stats['median']:.4f}")
        print(f"  mean:   {stats['mean']:.4f}")
        print(f"  q3:     {stats['q3']:.4f}")
        print(f"  max:    {stats['max']:.4f}")
        print()

    print(f"Salvato quality report in: {QUALITY_CSV}")
    print(f"Salvato summary in:        {SUMMARY_JSON}")
    print(f"Salvati grafici in:        {PLOTS_DIR}")


if __name__ == "__main__":
    main()