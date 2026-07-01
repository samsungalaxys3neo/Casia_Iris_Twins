from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata_clean.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "CASIA-Iris-Twins"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "quality"


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


def read_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Read metadata while preserving important identifier columns as strings.
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
    return pd.read_csv(metadata_path, dtype=dtype_map)


def get_image_path(row: pd.Series, dataset_root: Path) -> Path:
    """
    Build the image path from the portable relative_path column.
    If the dataset is inside the project and project_relative_path exists,
    this can also be used as fallback.
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

    # Return the most informative expected path, even if it does not exist.
    if pd.notna(relative_path) and str(relative_path).strip():
        return dataset_root / str(relative_path)

    raise ValueError("No relative_path or project_relative_path available for this row.")


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
            "error": f"cv2.imread failed: {image_path}",
        }

    img_float = img.astype(np.float32)

    brightness = float(np.mean(img_float))
    contrast = float(np.std(img_float))

    # Varianza del Laplaciano: più è alta, più l'immagine tende a essere nitida.
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
    plt.boxplot(groups, tick_labels=labels)
    plt.title(f"{column} by folder class")
    plt.xlabel("folder class")
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_metric(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
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


def run_quality_audit(
    metadata_path: Path,
    dataset_root: Path,
    output_dir: Path,
) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata non trovato: {metadata_path}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")

    quality_csv = output_dir / "quality_report.csv"
    summary_json = output_dir / "quality_summary.json"
    plots_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = read_metadata(metadata_path)

    rows = []

    print("=== QUALITY AUDIT ===")
    print(f"Metadata: {project_display_path(metadata_path)}")
    print(f"Dataset root: {project_display_path(dataset_root)}")
    print(f"Immagini da processare: {len(df)}")
    print()

    for idx, row in df.iterrows():
        try:
            image_path = get_image_path(row, dataset_root)
            quality = compute_quality(image_path)
        except Exception as e:
            quality = {
                "quality_ok": False,
                "brightness": None,
                "contrast": None,
                "focus_score": None,
                "dark_ratio": None,
                "bright_ratio": None,
                "error": str(e),
            }

        output_row = row.to_dict()
        output_row.update(quality)
        output_row["folder_class"] = f"{row['twin_id']}{row['eye']}"

        rows.append(output_row)

        if (idx + 1) % 250 == 0:
            print(f"Processate {idx + 1}/{len(df)} immagini")

    qdf = pd.DataFrame(rows)
    qdf.to_csv(quality_csv, index=False)

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
        "metadata_path": project_display_path(metadata_path),
        "dataset_root": project_display_path(dataset_root),
        "quality_csv": project_display_path(quality_csv),
        "plots_dir": project_display_path(plots_dir),
        "metrics": {},
    }

    for col in numeric_cols:
        summary["metrics"][col] = summarize_metric(qdf[col])

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for col in numeric_cols:
        save_histogram(qdf, col, plots_dir / f"hist_{col}.png")
        save_boxplot_by_eye(qdf, col, plots_dir / f"box_{col}_by_folder_class.png")

    print()
    print("=== QUALITY SUMMARY ===")
    print(f"Quality ok:     {summary['n_quality_ok']}")
    print(f"Quality failed: {summary['n_quality_failed']}")
    print()

    for col, stats in summary["metrics"].items():
        print(f"{col}")
        for key in ["min", "q1", "median", "mean", "q3", "max"]:
            value = stats[key]
            if value is None:
                print(f"  {key}: None")
            else:
                print(f"  {key}: {value:.4f}")
        print()

    print(f"Salvato quality report in: {project_display_path(quality_csv)}")
    print(f"Salvato summary in:        {project_display_path(summary_json)}")
    print(f"Salvati grafici in:        {project_display_path(plots_dir)}")

    return qdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path a metadata_clean.csv. Default: data/metadata/metadata_clean.csv",
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
        help="Cartella dove salvare il quality audit. Default: data/quality",
    )

    args = parser.parse_args()

    metadata_path = resolve_from_project(args.metadata)
    dataset_root = resolve_from_project(args.dataset_root)
    output_dir = resolve_from_project(args.output_dir)

    run_quality_audit(metadata_path, dataset_root, output_dir)


if __name__ == "__main__":
    main()
