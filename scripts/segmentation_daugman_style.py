from pathlib import Path
import argparse
import json
import math
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata_clean.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "CASIA-Iris-Twins"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "segmentation_daugman"

THUMB_SIZE = (180, 135)
LABEL_HEIGHT = 42
CONTACT_COLS = 5


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
    return pd.read_csv(metadata_path, dtype=dtype_map)


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


def safe_overlay_name(relative_path: str) -> str:
    return str(relative_path).replace("/", "__").replace("\\", "__") + ".png"


def preprocess(gray):
    """
    Preprocessing leggero:
    - blur per ridurre rumore;
    - CLAHE per migliorare contrasto locale.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    return enhanced


def gaussian_smooth_1d(x, sigma=1.5):
    radius = int(max(3, round(3 * sigma)))
    k = np.arange(-radius, radius + 1)
    g = np.exp(-(k ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return np.convolve(x, g, mode="same")


def sample_circle_mean(img, cx, cy, r, angles):
    h, w = img.shape

    xs = np.round(cx + r * np.cos(angles)).astype(np.int32)
    ys = np.round(cy + r * np.sin(angles)).astype(np.int32)

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    if valid.sum() < 0.70 * len(angles):
        return np.nan

    return float(np.mean(img[ys[valid], xs[valid]]))


def circular_profile(img, cx, cy, radii, angles):
    values = []

    for r in radii:
        values.append(sample_circle_mean(img, cx, cy, r, angles))

    values = np.array(values, dtype=np.float32)

    if np.isnan(values).all():
        return None

    if np.isnan(values).any():
        valid = ~np.isnan(values)
        values[~valid] = np.interp(
            np.flatnonzero(~valid),
            np.flatnonzero(valid),
            values[valid],
        )

    return values


def best_radius_by_daugman_operator(
    img,
    cx,
    cy,
    radii,
    angles,
    sigma=1.5,
    positive_only=True,
):
    """
    Versione semplificata dell'idea di Daugman:
    per ogni raggio calcoliamo la media lungo la circonferenza;
    poi cerchiamo dove la derivata radiale è massima.
    """
    profile = circular_profile(img, cx, cy, radii, angles)

    if profile is None or len(profile) < 5:
        return None

    smooth = gaussian_smooth_1d(profile, sigma=sigma)
    derivative = np.gradient(smooth)

    if positive_only:
        score_curve = derivative
    else:
        score_curve = np.abs(derivative)

    if len(score_curve) > 6:
        score_curve[:3] = -np.inf
        score_curve[-3:] = -np.inf

    idx = int(np.argmax(score_curve))
    best_score = float(score_curve[idx])

    return {
        "r": float(radii[idx]),
        "score": best_score,
        "profile_mean": float(profile[idx]),
        "profile_before": float(profile[max(0, idx - 2)]),
        "profile_after": float(profile[min(len(profile) - 1, idx + 2)]),
    }


def circle_mask(shape, cx, cy, r):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return mask


def mean_inside_circle(gray, cx, cy, r):
    mask = circle_mask(gray.shape, cx, cy, r)
    values = gray[mask > 0]

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def mean_in_ring(gray, cx, cy, r1, r2):
    outer = circle_mask(gray.shape, cx, cy, r2)
    inner = circle_mask(gray.shape, cx, cy, r1)
    ring = (outer > 0) & (inner == 0)
    values = gray[ring]

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def find_initial_pupil(gray, enhanced):
    """
    Trova un candidato iniziale per la pupilla usando:
    - regioni scure;
    - connected components;
    - fallback HoughCircles.
    """
    candidates = []
    h, w = gray.shape

    for p in [2, 3, 4, 5, 6, 8, 10]:
        thr = np.percentile(enhanced, p)
        thr = min(max(thr, 15), 90)

        dark = (enhanced <= thr).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)
            (cx, cy), r = cv2.minEnclosingCircle(cnt)

            if r < 12 or r > 90:
                continue

            if cx < r or cx > w - r or cy < r or cy > h - r:
                continue

            fill_ratio = area / (math.pi * r * r)

            inner = mean_inside_circle(gray, cx, cy, r * 0.80)
            ring = mean_in_ring(gray, cx, cy, r * 1.05, r * 1.55)

            if np.isnan(inner) or np.isnan(ring):
                continue

            darkness = 255 - inner
            contrast = max(ring - inner, 0)
            center_penalty = 15 * (abs(cx - w / 2) / w + abs(cy - h / 2) / h)

            score = (
                darkness
                + 2.0 * contrast
                + 20 * circularity
                + 10 * fill_ratio
                - center_penalty
            )

            candidates.append({
                "x": float(cx),
                "y": float(cy),
                "r": float(r),
                "score": float(score),
                "method": f"dark_components_p{p}",
            })

    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=90,
        param2=18,
        minRadius=12,
        maxRadius=90,
    )

    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            inner = mean_inside_circle(gray, cx, cy, r * 0.80)
            ring = mean_in_ring(gray, cx, cy, r * 1.05, r * 1.55)

            if np.isnan(inner) or np.isnan(ring):
                continue

            darkness = 255 - inner
            contrast = max(ring - inner, 0)
            score = darkness + 2.0 * contrast

            candidates.append({
                "x": float(cx),
                "y": float(cy),
                "r": float(r),
                "score": float(score),
                "method": "hough_pupil",
            })

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    return candidates[0]


def refine_pupil_daugman(gray, enhanced, init):
    """
    Raffina la pupilla con una ricerca Daugman-style locale.
    """
    full_angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)

    cx0, cy0, r0 = init["x"], init["y"], init["r"]

    best = None
    offsets = [-8, -4, 0, 4, 8]

    min_r = int(max(10, r0 - 18))
    max_r = int(min(95, r0 + 18))

    radii = np.arange(min_r, max_r + 1, 1)

    for dx in offsets:
        for dy in offsets:
            cx = cx0 + dx
            cy = cy0 + dy

            result = best_radius_by_daugman_operator(
                enhanced,
                cx,
                cy,
                radii,
                full_angles,
                sigma=1.3,
                positive_only=True,
            )

            if result is None:
                continue

            r = result["r"]

            inner = mean_inside_circle(gray, cx, cy, r * 0.80)
            ring = mean_in_ring(gray, cx, cy, r * 1.05, r * 1.50)

            if np.isnan(inner) or np.isnan(ring):
                continue

            darkness = 255 - inner
            contrast = max(ring - inner, 0)

            combined_score = (
                3.0 * result["score"]
                + 0.25 * darkness
                + 0.80 * contrast
                - 0.15 * math.hypot(dx, dy)
            )

            candidate = {
                "x": float(cx),
                "y": float(cy),
                "r": float(r),
                "score": float(combined_score),
                "daugman_score": float(result["score"]),
                "inner_mean": float(inner),
                "ring_mean": float(ring),
                "method": "daugman_pupil",
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    return best


def lateral_angles():
    """
    Per il limbus usiamo soprattutto lati sinistro/destro,
    perché sopra e sotto ci sono spesso palpebre/ciglia.
    """
    deg1 = np.linspace(-55, 55, 120)
    deg2 = np.linspace(125, 235, 120)
    return np.deg2rad(np.concatenate([deg1, deg2]))


def find_iris_daugman(gray, enhanced, pupil):
    """
    Localizza il limbus con ricerca Daugman-style.
    """
    angles = lateral_angles()

    px, py, pr = pupil["x"], pupil["y"], pupil["r"]

    min_r = int(max(75, pr * 1.9))
    max_r = int(min(210, pr * 4.3))

    if max_r <= min_r + 10:
        return None

    radii = np.arange(min_r, max_r + 1, 2)

    best = None
    offsets = [-20, -12, -6, 0, 6, 12, 20]

    for dx in offsets:
        for dy in offsets:
            cx = px + dx
            cy = py + dy

            result = best_radius_by_daugman_operator(
                enhanced,
                cx,
                cy,
                radii,
                angles,
                sigma=2.0,
                positive_only=True,
            )

            if result is None:
                continue

            r = result["r"]
            ratio = r / pr if pr > 0 else np.nan
            center_offset = math.hypot(cx - px, cy - py)

            if np.isnan(ratio):
                continue

            if ratio < 1.8 or ratio > 5.0:
                continue

            if center_offset > 35:
                continue

            inside = mean_in_ring(gray, cx, cy, max(pr + 5, r - 12), r - 3)
            outside = mean_in_ring(gray, cx, cy, r + 3, min(r + 16, r + 25))

            if np.isnan(inside) or np.isnan(outside):
                continue

            contrast = outside - inside

            ratio_penalty = 2.0 * abs(ratio - 2.55)
            offset_penalty = 0.12 * center_offset

            combined_score = (
                4.0 * result["score"]
                + 0.80 * max(contrast, 0)
                - ratio_penalty
                - offset_penalty
            )

            candidate = {
                "x": float(cx),
                "y": float(cy),
                "r": float(r),
                "score": float(combined_score),
                "daugman_score": float(result["score"]),
                "contrast": float(contrast),
                "ratio": float(ratio),
                "center_offset": float(center_offset),
                "method": "daugman_limbus_lateral",
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        return None

    if best["score"] >= 45:
        best["confidence"] = "high"
    elif best["score"] >= 28:
        best["confidence"] = "medium"
    else:
        best["confidence"] = "low"

    return best


def segment_one(image_path: Path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        return {
            "segmentation_ok": False,
            "pupil_found": False,
            "iris_found": False,
            "error": f"cv2.imread failed: {image_path}",
        }, None

    enhanced = preprocess(gray)

    init = find_initial_pupil(gray, enhanced)

    if init is None:
        return {
            "segmentation_ok": False,
            "pupil_found": False,
            "iris_found": False,
            "error": "initial pupil not found",
        }, gray

    pupil = refine_pupil_daugman(gray, enhanced, init)

    if pupil is None:
        return {
            "segmentation_ok": False,
            "pupil_found": False,
            "iris_found": False,
            "error": "daugman pupil refinement failed",
        }, gray

    iris = find_iris_daugman(gray, enhanced, pupil)

    if iris is None:
        return {
            "segmentation_ok": False,
            "pupil_found": True,
            "iris_found": False,
            "pupil_x": pupil["x"],
            "pupil_y": pupil["y"],
            "pupil_r": pupil["r"],
            "pupil_score": pupil["score"],
            "pupil_daugman_score": pupil["daugman_score"],
            "pupil_inner_mean": pupil["inner_mean"],
            "pupil_ring_mean": pupil["ring_mean"],
            "error": "iris limbus not found",
        }, gray

    ok = (
        pupil["inner_mean"] <= 95
        and pupil["r"] >= 18
        and pupil["r"] <= 85
        and iris["confidence"] in {"medium", "high"}
        and 2.0 <= iris["ratio"] <= 4.2
        and iris["center_offset"] <= 30
    )

    return {
        "segmentation_ok": bool(ok),
        "pupil_found": True,
        "iris_found": True,

        "pupil_x": pupil["x"],
        "pupil_y": pupil["y"],
        "pupil_r": pupil["r"],
        "pupil_score": pupil["score"],
        "pupil_daugman_score": pupil["daugman_score"],
        "pupil_inner_mean": pupil["inner_mean"],
        "pupil_ring_mean": pupil["ring_mean"],
        "pupil_method": pupil["method"],

        "iris_x": iris["x"],
        "iris_y": iris["y"],
        "iris_r": iris["r"],
        "iris_score": iris["score"],
        "iris_daugman_score": iris["daugman_score"],
        "iris_contrast": iris["contrast"],
        "iris_confidence": iris["confidence"],
        "iris_pupil_radius_ratio": iris["ratio"],
        "iris_center_offset": iris["center_offset"],
        "iris_method": iris["method"],

        "error": "",
    }, gray


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
        cv2.circle(rgb, (ix, iy), 2, (0, 255, 0), -1)

    status = "OK" if result.get("segmentation_ok", False) else "CHECK"
    conf = result.get("iris_confidence", "none")
    ratio = result.get("iris_pupil_radius_ratio", np.nan)

    if not np.isnan(ratio):
        text = f"{status} | conf={conf} | ratio={ratio:.2f} | {label}"
    else:
        text = f"{status} | conf={conf} | {label}"

    cv2.rectangle(rgb, (0, 0), (rgb.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(
        rgb,
        text[:95],
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return rgb


def save_contact_sheet(paths, title, output_path):
    font = ImageFont.load_default()
    thumbs = []

    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(THUMB_SIZE)
        except Exception:
            continue

        canvas = Image.new("RGB", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), "white")
        x = (THUMB_SIZE[0] - img.width) // 2
        canvas.paste(img, (x, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((4, THUMB_SIZE[1] + 4), path.name[:32], fill="black", font=font)

        thumbs.append(canvas)

    if not thumbs:
        return

    rows = math.ceil(len(thumbs) / CONTACT_COLS)

    sheet = Image.new(
        "RGB",
        (CONTACT_COLS * THUMB_SIZE[0], rows * (THUMB_SIZE[1] + LABEL_HEIGHT) + 35),
        "white",
    )

    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black", font=font)

    y0 = 35

    for i, thumb in enumerate(thumbs):
        x = (i % CONTACT_COLS) * THUMB_SIZE[0]
        y = y0 + (i // CONTACT_COLS) * (THUMB_SIZE[1] + LABEL_HEIGHT)
        sheet.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def summarize_numeric_column(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None

    values = pd.to_numeric(df[col], errors="coerce").dropna()

    if len(values) == 0:
        return None

    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def run_daugman_segmentation(
    metadata_path: Path,
    dataset_root: Path,
    output_dir: Path,
    limit,
    debug_overlays: int,
):
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata non trovato: {metadata_path}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")

    overlay_dir = output_dir / "overlays"
    contact_dir = output_dir / "contact_sheets"
    report_path = output_dir / "segmentation_daugman_report.csv"
    summary_path = output_dir / "segmentation_daugman_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    df = read_metadata(metadata_path)

    if limit is not None:
        df = df.head(limit).copy()

    print("=== DAUGMAN-STYLE SEGMENTATION ===")
    print(f"Metadata: {project_display_path(metadata_path)}")
    print(f"Dataset root: {project_display_path(dataset_root)}")
    print(f"Output dir: {project_display_path(output_dir)}")
    print(f"Immagini da segmentare: {len(df)}")
    print()

    rows = []

    ok_paths = []
    high_paths = []
    medium_paths = []
    check_paths = []
    fail_paths = []

    ok_saved = 0

    for processed_idx, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            image_path = get_image_path(row, dataset_root)
            result, gray = segment_one(image_path)
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

        save_overlay = False

        if result.get("segmentation_ok", False):
            if ok_saved < debug_overlays:
                save_overlay = True
                ok_saved += 1
        else:
            save_overlay = True

        if gray is not None and save_overlay:
            relative_path = str(row["relative_path"])
            overlay = draw_overlay(gray, result, relative_path)
            overlay_path = overlay_dir / safe_overlay_name(relative_path)
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if result.get("segmentation_ok", False):
                ok_paths.append(overlay_path)

                if result.get("iris_confidence") == "high":
                    high_paths.append(overlay_path)
                elif result.get("iris_confidence") == "medium":
                    medium_paths.append(overlay_path)
            else:
                check_paths.append(overlay_path)

                if not result.get("pupil_found", False) or not result.get("iris_found", False):
                    fail_paths.append(overlay_path)

        if processed_idx % 100 == 0:
            print(f"Processate {processed_idx}/{len(df)} immagini")

    sdf = pd.DataFrame(rows)
    sdf.to_csv(report_path, index=False)

    n = len(sdf)
    n_ok = int(sdf["segmentation_ok"].sum())
    n_check = n - n_ok
    pupil_found = int(sdf["pupil_found"].sum())
    iris_found = int(sdf["iris_found"].sum())

    summary = {
        "n_images": n,
        "n_segmentation_ok": n_ok,
        "n_segmentation_check_or_fail": n_check,
        "segmentation_ok_rate": n_ok / n if n else None,
        "pupil_found": pupil_found,
        "iris_found": iris_found,
        "metadata_path": project_display_path(metadata_path),
        "dataset_root": project_display_path(dataset_root),
        "output_dir": project_display_path(output_dir),
    }

    if "iris_confidence" in sdf.columns:
        summary["iris_confidence_counts"] = (
            sdf["iris_confidence"].fillna("none").value_counts().to_dict()
        )

    for col in [
        "pupil_r",
        "pupil_score",
        "pupil_daugman_score",
        "iris_r",
        "iris_score",
        "iris_daugman_score",
        "iris_pupil_radius_ratio",
        "iris_center_offset",
    ]:
        stats = summarize_numeric_column(sdf, col)
        if stats is not None:
            summary[col] = stats

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    random.seed(42)

    def sample_paths(paths, k=25):
        if len(paths) <= k:
            return paths
        return random.sample(paths, k)

    save_contact_sheet(
        sample_paths(ok_paths, 25),
        "Daugman-style OK examples",
        contact_dir / "daugman_ok_examples.png",
    )

    save_contact_sheet(
        sample_paths(high_paths, 25),
        "Daugman-style HIGH confidence examples",
        contact_dir / "daugman_high_confidence_examples.png",
    )

    save_contact_sheet(
        sample_paths(medium_paths, 25),
        "Daugman-style MEDIUM confidence examples",
        contact_dir / "daugman_medium_confidence_examples.png",
    )

    save_contact_sheet(
        sample_paths(check_paths, 25),
        "Daugman-style CHECK examples",
        contact_dir / "daugman_check_examples.png",
    )

    save_contact_sheet(
        sample_paths(fail_paths, 25),
        "Daugman-style FAIL examples",
        contact_dir / "daugman_fail_examples.png",
    )

    print()
    print("=== DAUGMAN-STYLE SEGMENTATION SUMMARY ===")
    print(f"Immagini totali:       {n}")
    print(f"Segmentation OK:       {n_ok}")
    print(f"Check/fail:            {n_check}")
    print(f"OK rate:               {n_ok / n if n else 0:.4f}")
    print(f"Pupil found:           {pupil_found}")
    print(f"Iris found:            {iris_found}")

    if "iris_confidence_counts" in summary:
        print()
        print("Iris confidence:")
        for k, v in summary["iris_confidence_counts"].items():
            print(f"  {k}: {v}")

    print()

    for col in ["pupil_r", "iris_r", "iris_pupil_radius_ratio", "iris_center_offset"]:
        if col in summary:
            print(col)
            for k, v in summary[col].items():
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
        help="Path a metadata CSV. Default: data/metadata/metadata_clean.csv",
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
        help="Cartella dove salvare i risultati. Default: data/segmentation_daugman",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Processa solo le prime N immagini",
    )
    parser.add_argument(
        "--debug-overlays",
        type=int,
        default=400,
        help="Numero massimo overlay OK da salvare",
    )

    args = parser.parse_args()

    metadata_path = resolve_from_project(args.metadata)
    dataset_root = resolve_from_project(args.dataset_root)
    output_dir = resolve_from_project(args.output_dir)

    run_daugman_segmentation(
        metadata_path=metadata_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        limit=args.limit,
        debug_overlays=args.debug_overlays,
    )


if __name__ == "__main__":
    main()
