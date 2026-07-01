from pathlib import Path
import argparse
import math
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata_clean.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "CASIA-Iris-Twins"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "segmentation"

DEFAULT_N_DEBUG_OVERLAYS = 300
CONTACT_SHEET_COLS = 5
THUMB_SIZE = (180, 135)
LABEL_HEIGHT = 38


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


def find_pupil(gray):
    """
    Localizzazione semplice della pupilla:
    - la pupilla è una regione molto scura;
    - cerchiamo blob scuri abbastanza circolari;
    - scegliamo il candidato migliore.
    """
    h, w = gray.shape

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    q = np.percentile(blur, 8)
    threshold = min(70, max(25, q + 5))

    dark = (blur < threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)

        (x, y), r = cv2.minEnclosingCircle(cnt)
        x, y, r = float(x), float(y), float(r)

        if r < 12 or r > 95:
            continue

        if x < 40 or x > w - 40 or y < 40 or y > h - 40:
            continue

        center_penalty = abs(x - w / 2) / w + abs(y - h / 2) / h
        score = area * max(circularity, 0.01) / (1 + center_penalty)

        candidates.append({
            "x": x,
            "y": y,
            "r": r,
            "area": area,
            "circularity": circularity,
            "score": score,
            "threshold": threshold,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0]


def circle_mean_intensity(gray, cx, cy, r, n_points=360):
    h, w = gray.shape
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    xs = np.round(cx + r * np.cos(angles)).astype(np.int32)
    ys = np.round(cy + r * np.sin(angles)).astype(np.int32)

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    if valid.sum() < n_points * 0.65:
        return None

    return float(np.mean(gray[ys[valid], xs[valid]]))


def find_iris_radius(gray, pupil):
    """
    Stima preliminare del raggio dell'iride.
    Usiamo il centro della pupilla e cerchiamo il raggio in cui
    l'intensità media lungo la circonferenza cambia maggiormente.
    """
    h, w = gray.shape
    cx, cy, pr = pupil["x"], pupil["y"], pupil["r"]

    min_r = int(max(pr * 1.8, pr + 35, 65))
    max_r = int(min(pr * 4.2, 210, cx - 5, cy - 5, w - cx - 5, h - cy - 5))

    if max_r <= min_r + 20:
        return None

    radii = list(range(min_r, max_r + 1))
    means = []

    for r in radii:
        m = circle_mean_intensity(gray, cx, cy, r)
        means.append(np.nan if m is None else m)

    means = np.array(means, dtype=np.float32)

    if np.isnan(means).mean() > 0.3:
        return None

    valid = ~np.isnan(means)
    means[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), means[valid])

    means_smooth = cv2.GaussianBlur(means.reshape(1, -1), (1, 9), 0).flatten()

    grad = np.gradient(means_smooth)

    best_idx = int(np.argmax(grad))
    best_r = radii[best_idx]
    best_grad = float(grad[best_idx])

    if best_grad < 0.15:
        confidence = "low"
    elif best_grad < 0.45:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "x": cx,
        "y": cy,
        "r": float(best_r),
        "gradient": best_grad,
        "confidence": confidence,
        "min_r": min_r,
        "max_r": max_r,
    }


def segment_image(image_path: Path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        return {
            "segmentation_ok": False,
            "pupil_found": False,
            "iris_found": False,
            "error": f"cv2.imread failed: {image_path}",
        }, None

    pupil = find_pupil(gray)

    if pupil is None:
        return {
            "segmentation_ok": False,
            "pupil_found": False,
            "iris_found": False,
            "error": "pupil not found",
        }, gray

    iris = find_iris_radius(gray, pupil)

    if iris is None:
        return {
            "segmentation_ok": False,
            "pupil_found": True,
            "iris_found": False,
            "pupil_x": pupil["x"],
            "pupil_y": pupil["y"],
            "pupil_r": pupil["r"],
            "pupil_circularity": pupil["circularity"],
            "pupil_threshold": pupil["threshold"],
            "error": "iris radius not found",
        }, gray

    ratio = iris["r"] / pupil["r"] if pupil["r"] > 0 else None

    segmentation_ok = (
        ratio is not None
        and 1.8 <= ratio <= 5.0
        and pupil["circularity"] >= 0.35
    )

    result = {
        "segmentation_ok": bool(segmentation_ok),
        "pupil_found": True,
        "iris_found": True,

        "pupil_x": pupil["x"],
        "pupil_y": pupil["y"],
        "pupil_r": pupil["r"],
        "pupil_circularity": pupil["circularity"],
        "pupil_threshold": pupil["threshold"],

        "iris_x": iris["x"],
        "iris_y": iris["y"],
        "iris_r": iris["r"],
        "iris_gradient": iris["gradient"],
        "iris_confidence": iris["confidence"],

        "iris_pupil_radius_ratio": ratio,
        "error": "",
    }

    return result, gray


def draw_overlay(gray, result, label):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if result.get("pupil_found", False):
        px = int(round(result["pupil_x"]))
        py = int(round(result["pupil_y"]))
        pr = int(round(result["pupil_r"]))
        cv2.circle(rgb, (px, py), pr, (255, 0, 0), 2)
        cv2.circle(rgb, (px, py), 2, (255, 0, 0), -1)

    if result.get("iris_found", False):
        ix = int(round(result["iris_x"]))
        iy = int(round(result["iris_y"]))
        ir = int(round(result["iris_r"]))
        cv2.circle(rgb, (ix, iy), ir, (0, 255, 0), 2)

    status = "OK" if result.get("segmentation_ok", False) else "CHECK"
    text = f"{status} | {label}"

    cv2.rectangle(rgb, (0, 0), (rgb.shape[1], 28), (255, 255, 255), -1)
    cv2.putText(
        rgb,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return rgb


def save_contact_sheet(image_paths, title, output_path):
    font = ImageFont.load_default()
    thumbs = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(THUMB_SIZE)
        except Exception:
            continue

        canvas = Image.new("RGB", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), "white")
        x = (THUMB_SIZE[0] - img.width) // 2
        canvas.paste(img, (x, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((4, THUMB_SIZE[1] + 4), path.name[:30], fill="black", font=font)

        thumbs.append(canvas)

    if not thumbs:
        return

    rows = math.ceil(len(thumbs) / CONTACT_SHEET_COLS)
    sheet = Image.new(
        "RGB",
        (CONTACT_SHEET_COLS * THUMB_SIZE[0], rows * (THUMB_SIZE[1] + LABEL_HEIGHT) + 35),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black", font=font)

    y0 = 35

    for i, thumb in enumerate(thumbs):
        x = (i % CONTACT_SHEET_COLS) * THUMB_SIZE[0]
        y = y0 + (i // CONTACT_SHEET_COLS) * (THUMB_SIZE[1] + LABEL_HEIGHT)
        sheet.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def run_preliminary_segmentation(
    metadata_path: Path,
    dataset_root: Path,
    output_dir: Path,
    n_debug_overlays: int,
):
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata non trovato: {metadata_path}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")

    overlay_dir = output_dir / "overlays"
    contact_dir = output_dir / "contact_sheets"
    report_path = output_dir / "segmentation_report.csv"
    summary_path = output_dir / "segmentation_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    df = read_metadata(metadata_path)

    print("=== PRELIMINARY SEGMENTATION ===")
    print(f"Metadata: {project_display_path(metadata_path)}")
    print(f"Dataset root: {project_display_path(dataset_root)}")
    print(f"Output dir: {project_display_path(output_dir)}")
    print(f"Immagini da segmentare: {len(df)}")
    print()

    rows = []
    overlay_paths_ok = []
    overlay_paths_check = []

    for idx, row in df.iterrows():
        try:
            image_path = get_image_path(row, dataset_root)
            result, gray = segment_image(image_path)
        except Exception as e:
            result = {
                "segmentation_ok": False,
                "pupil_found": False,
                "iris_found": False,
                "error": str(e),
            }
            gray = None

        out_row = row.to_dict()
        out_row.update(result)
        rows.append(out_row)

        save_debug = idx < n_debug_overlays or not result.get("segmentation_ok", False)

        if gray is not None and save_debug:
            label = str(row["relative_path"])
            overlay = draw_overlay(gray, result, label)

            safe_name = str(row["relative_path"]).replace("/", "__")
            overlay_path = overlay_dir / f"{safe_name}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if result.get("segmentation_ok", False):
                overlay_paths_ok.append(overlay_path)
            else:
                overlay_paths_check.append(overlay_path)

        if (idx + 1) % 250 == 0:
            print(f"Processate {idx + 1}/{len(df)} immagini")

    sdf = pd.DataFrame(rows)
    sdf.to_csv(report_path, index=False)

    n = len(sdf)
    n_ok = int(sdf["segmentation_ok"].sum())
    n_fail = n - n_ok

    summary = {
        "n_images": n,
        "n_segmentation_ok": n_ok,
        "n_segmentation_check_or_fail": n_fail,
        "segmentation_ok_rate": n_ok / n if n > 0 else None,
        "pupil_found": int(sdf["pupil_found"].sum()),
        "iris_found": int(sdf["iris_found"].sum()),
        "metadata_path": project_display_path(metadata_path),
        "dataset_root": project_display_path(dataset_root),
        "output_dir": project_display_path(output_dir),
    }

    if "iris_pupil_radius_ratio" in sdf.columns:
        ratio_values = pd.to_numeric(sdf["iris_pupil_radius_ratio"], errors="coerce").dropna()
        summary["iris_pupil_radius_ratio"] = {
            "min": float(ratio_values.min()) if len(ratio_values) else None,
            "median": float(ratio_values.median()) if len(ratio_values) else None,
            "mean": float(ratio_values.mean()) if len(ratio_values) else None,
            "max": float(ratio_values.max()) if len(ratio_values) else None,
        }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_contact_sheet(
        overlay_paths_ok[:25],
        "Segmentation examples - OK",
        contact_dir / "segmentation_ok_examples.png",
    )

    save_contact_sheet(
        overlay_paths_check[:25],
        "Segmentation examples - CHECK",
        contact_dir / "segmentation_check_examples.png",
    )

    print()
    print("=== SEGMENTATION SUMMARY ===")
    print(f"Immagini totali:       {n}")
    print(f"Segmentation OK:       {n_ok}")
    print(f"Check/fail:            {n_fail}")

    if summary["segmentation_ok_rate"] is not None:
        print(f"OK rate:               {summary['segmentation_ok_rate']:.4f}")

    print(f"Pupil found:           {summary['pupil_found']}")
    print(f"Iris found:            {summary['iris_found']}")

    if "iris_pupil_radius_ratio" in summary:
        print()
        print("iris_pupil_radius_ratio:")
        for k, v in summary["iris_pupil_radius_ratio"].items():
            print(f"  {k}: {v}")

    print()
    print(f"Report salvato in:     {project_display_path(report_path)}")
    print(f"Summary salvato in:    {project_display_path(summary_path)}")
    print(f"Overlay salvati in:    {project_display_path(overlay_dir)}")
    print(f"Contact sheet in:      {project_display_path(contact_dir)}")


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
        help="Cartella dove salvare i risultati. Default: data/segmentation",
    )
    parser.add_argument(
        "--n-debug-overlays",
        type=int,
        default=DEFAULT_N_DEBUG_OVERLAYS,
        help="Numero di overlay iniziali da salvare. Default: 300",
    )

    args = parser.parse_args()

    metadata_path = resolve_from_project(args.metadata)
    dataset_root = resolve_from_project(args.dataset_root)
    output_dir = resolve_from_project(args.output_dir)

    run_preliminary_segmentation(
        metadata_path=metadata_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        n_debug_overlays=args.n_debug_overlays,
    )


if __name__ == "__main__":
    main()
