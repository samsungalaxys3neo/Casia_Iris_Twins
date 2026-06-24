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

METADATA_PATH = PROJECT_ROOT / "data/metadata/metadata_final.csv"
FALLBACK_METADATA_PATH = PROJECT_ROOT / "data/metadata/metadata_clean.csv"

OUTPUT_DIR = PROJECT_ROOT / "data/segmentation_v2"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
CONTACT_DIR = OUTPUT_DIR / "contact_sheets"
REPORT_PATH = OUTPUT_DIR / "segmentation_v2_report.csv"
SUMMARY_PATH = OUTPUT_DIR / "segmentation_v2_summary.json"

THUMB_SIZE = (180, 135)
LABEL_HEIGHT = 40
CONTACT_COLS = 5


def preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.medianBlur(enhanced, 5)
    return enhanced


def circle_mask(shape, cx, cy, r):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return mask


def mean_inside_circle(gray, cx, cy, r):
    mask = circle_mask(gray.shape, cx, cy, r)
    values = gray[mask > 0]
    if len(values) == 0:
        return None
    return float(values.mean())


def mean_in_ring(gray, cx, cy, r_inner, r_outer):
    outer = circle_mask(gray.shape, cx, cy, r_outer)
    inner = circle_mask(gray.shape, cx, cy, r_inner)
    ring = (outer > 0) & (inner == 0)
    values = gray[ring]
    if len(values) == 0:
        return None
    return float(values.mean())


def sample_circle_values(img, cx, cy, r, angles_rad):
    h, w = img.shape

    xs = np.round(cx + r * np.cos(angles_rad)).astype(np.int32)
    ys = np.round(cy + r * np.sin(angles_rad)).astype(np.int32)

    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    if valid.sum() < 0.65 * len(angles_rad):
        return None

    return img[ys[valid], xs[valid]].astype(np.float32)


def lateral_angles():
    # Usiamo soprattutto i lati dell'iride, perché sopra/sotto spesso ci sono palpebre.
    deg1 = np.linspace(-45, 45, 100)
    deg2 = np.linspace(135, 225, 100)
    return np.deg2rad(np.concatenate([deg1, deg2]))


def all_angles():
    return np.linspace(0, 2 * np.pi, 260, endpoint=False)


def score_pupil_candidate(gray, cx, cy, r, circularity=1.0, fill_ratio=1.0):
    h, w = gray.shape

    if r < 12 or r > 85:
        return None

    if cx < r + 5 or cx > w - r - 5 or cy < r + 5 or cy > h - r - 5:
        return None

    inner_mean = mean_inside_circle(gray, cx, cy, r * 0.80)
    ring_mean = mean_in_ring(gray, cx, cy, r * 1.05, r * 1.55)

    if inner_mean is None or ring_mean is None:
        return None

    # La pupilla deve essere scura e il bordo/ring attorno tendenzialmente più chiaro.
    darkness_score = 255.0 - inner_mean
    contrast_score = max(ring_mean - inner_mean, 0.0)

    # Penalizziamo candidati troppo lontani dal centro immagine, ma in modo leggero.
    center_penalty = 25.0 * (abs(cx - w / 2) / w + abs(cy - h / 2) / h)

    score = (
        darkness_score
        + 2.0 * contrast_score
        + 25.0 * circularity
        + 15.0 * fill_ratio
        - center_penalty
    )

    return {
        "x": float(cx),
        "y": float(cy),
        "r": float(r),
        "score": float(score),
        "inner_mean": float(inner_mean),
        "ring_mean": float(ring_mean),
        "circularity": float(circularity),
        "fill_ratio": float(fill_ratio),
    }


def find_pupil_candidates(gray, enhanced):
    candidates = []
    h, w = gray.shape

    # Metodo 1: sogliatura sui pixel scuri + componenti connesse.
    percentiles = [2, 3, 4, 5, 6, 8, 10, 12]

    for p in percentiles:
        thr = np.percentile(enhanced, p)
        thr = min(max(thr, 20), 85)

        dark = (enhanced <= thr).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 250:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)
            (cx, cy), r = cv2.minEnclosingCircle(cnt)

            if r <= 0:
                continue

            fill_ratio = area / (math.pi * r * r)

            if circularity < 0.25 or fill_ratio < 0.25:
                continue

            cand = score_pupil_candidate(
                gray,
                cx,
                cy,
                r,
                circularity=circularity,
                fill_ratio=fill_ratio,
            )

            if cand is not None:
                cand["method"] = f"threshold_p{p}"
                candidates.append(cand)

    # Metodo 2: fallback HoughCircles.
    blur = cv2.GaussianBlur(enhanced, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=80,
        param2=18,
        minRadius=12,
        maxRadius=85,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        for cx, cy, r in circles:
            cand = score_pupil_candidate(
                gray,
                cx,
                cy,
                r,
                circularity=0.75,
                fill_ratio=0.75,
            )

            if cand is not None:
                cand["method"] = "hough_pupil"
                candidates.append(cand)

    return candidates


def deduplicate_candidates(candidates, dist_thr=8, radius_thr=8):
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept = []

    for c in candidates:
        duplicate = False

        for k in kept:
            dist = math.hypot(c["x"] - k["x"], c["y"] - k["y"])
            rdiff = abs(c["r"] - k["r"])

            if dist < dist_thr and rdiff < radius_thr:
                duplicate = True
                break

        if not duplicate:
            kept.append(c)

    return kept


def find_pupil(gray, enhanced):
    candidates = find_pupil_candidates(gray, enhanced)
    candidates = deduplicate_candidates(candidates)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    return candidates[0]


def compute_gradient(enhanced):
    sx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sx, sy)
    return grad


def iris_candidate_score(gray, grad, cx, cy, r, pupil):
    h, w = gray.shape

    if r < 60 or r > 220:
        return None

    if cx - r < 2 or cx + r >= w - 2 or cy - r < 2 or cy + r >= h - 2:
        return None

    pr = pupil["r"]
    ratio = r / pr if pr > 0 else None

    if ratio is None or ratio < 1.7 or ratio > 5.4:
        return None

    center_offset = math.hypot(cx - pupil["x"], cy - pupil["y"])
    if center_offset > max(35, 0.35 * pr):
        return None

    angles = lateral_angles()

    edge_vals = sample_circle_values(grad, cx, cy, r, angles)
    inside_vals = sample_circle_values(gray, cx, cy, r - 4, angles)
    outside_vals = sample_circle_values(gray, cx, cy, r + 4, angles)

    if edge_vals is None or inside_vals is None or outside_vals is None:
        return None

    edge_score = float(np.mean(edge_vals))
    contrast = float(np.mean(outside_vals) - np.mean(inside_vals))

    # Preferiamo transizione iride -> sclera: esterno più chiaro dell'interno.
    positive_contrast = max(contrast, 0.0)

    # Rapporto tipico nel dataset visto dalla v1: circa 2.6-2.8.
    ratio_penalty = 4.0 * abs(ratio - 2.75)
    offset_penalty = 0.25 * center_offset

    score = edge_score + 1.4 * positive_contrast - ratio_penalty - offset_penalty

    return {
        "x": float(cx),
        "y": float(cy),
        "r": float(r),
        "score": float(score),
        "edge_score": float(edge_score),
        "contrast": float(contrast),
        "ratio": float(ratio),
        "center_offset": float(center_offset),
    }


def find_iris(gray, enhanced, pupil):
    grad = compute_gradient(enhanced)

    px, py, pr = pupil["x"], pupil["y"], pupil["r"]

    min_r = int(max(70, pr * 1.9))
    max_r = int(min(210, pr * 5.1))

    candidates = []

    # Ricerca locale del centro dell'iride attorno alla pupilla.
    offsets = [-24, -16, -8, 0, 8, 16, 24]
    radius_step = 2

    for dy in offsets:
        for dx in offsets:
            cx = px + dx
            cy = py + dy

            for r in range(min_r, max_r + 1, radius_step):
                cand = iris_candidate_score(gray, grad, cx, cy, r, pupil)
                if cand is not None:
                    cand["method"] = "radial_lateral_search"
                    candidates.append(cand)

    # Fallback Hough per cerchi grandi.
    blur = cv2.GaussianBlur(enhanced, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.4,
        minDist=120,
        param1=80,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        for cx, cy, r in circles:
            cand = iris_candidate_score(gray, grad, cx, cy, r, pupil)
            if cand is not None:
                cand["method"] = "hough_iris"
                candidates.append(cand)

    candidates = deduplicate_candidates(candidates, dist_thr=12, radius_thr=10)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    if best["score"] >= 35:
        confidence = "high"
    elif best["score"] >= 24:
        confidence = "medium"
    else:
        confidence = "low"

    best["confidence"] = confidence
    return best


def segment_one(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        return {
            "segmentation_ok": False,
            "pupil_found": False,
            "iris_found": False,
            "error": "cv2.imread failed",
        }, None

    enhanced = preprocess(gray)

    pupil = find_pupil(gray, enhanced)

    if pupil is None:
        return {
            "segmentation_ok": False,
            "pupil_found": False,
            "iris_found": False,
            "error": "pupil not found",
        }, gray

    iris = find_iris(gray, enhanced, pupil)

    if iris is None:
        return {
            "segmentation_ok": False,
            "pupil_found": True,
            "iris_found": False,
            "pupil_x": pupil["x"],
            "pupil_y": pupil["y"],
            "pupil_r": pupil["r"],
            "pupil_score": pupil["score"],
            "pupil_method": pupil["method"],
            "pupil_inner_mean": pupil["inner_mean"],
            "pupil_ring_mean": pupil["ring_mean"],
            "pupil_circularity": pupil["circularity"],
            "error": "iris not found",
        }, gray

    # Criteri di accettazione più severi rispetto alla v1.
    ok = (
        pupil["inner_mean"] <= 90
        and pupil["score"] >= 140
        and iris["confidence"] in {"medium", "high"}
        and 1.8 <= iris["ratio"] <= 5.0
        and iris["center_offset"] <= 35
    )

    return {
        "segmentation_ok": bool(ok),
        "pupil_found": True,
        "iris_found": True,

        "pupil_x": pupil["x"],
        "pupil_y": pupil["y"],
        "pupil_r": pupil["r"],
        "pupil_score": pupil["score"],
        "pupil_method": pupil["method"],
        "pupil_inner_mean": pupil["inner_mean"],
        "pupil_ring_mean": pupil["ring_mean"],
        "pupil_circularity": pupil["circularity"],

        "iris_x": iris["x"],
        "iris_y": iris["y"],
        "iris_r": iris["r"],
        "iris_score": iris["score"],
        "iris_method": iris["method"],
        "iris_edge_score": iris["edge_score"],
        "iris_contrast": iris["contrast"],
        "iris_confidence": iris["confidence"],
        "iris_pupil_radius_ratio": iris["ratio"],
        "iris_center_offset": iris["center_offset"],

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


def safe_overlay_name(relative_path):
    return relative_path.replace("/", "__").replace("\\", "__") + ".png"


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

    sheet.save(output_path)


def load_metadata():
    path = METADATA_PATH if METADATA_PATH.exists() else FALLBACK_METADATA_PATH

    df = pd.read_csv(
        path,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        },
    )

    print(f"Uso metadata: {path}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Processa solo le prime N immagini")
    parser.add_argument("--debug-overlays", type=int, default=400, help="Numero massimo overlay OK da salvare")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("=== SEGMENTATION V2 ===")
    print(f"Immagini da segmentare: {len(df)}")
    print(f"Output: {OUTPUT_DIR}")

    rows = []

    ok_overlay_paths = []
    check_overlay_paths = []
    fail_overlay_paths = []
    medium_overlay_paths = []
    high_overlay_paths = []

    ok_saved = 0

    for idx, row in df.iterrows():
        image_path = Path(row["absolute_path"])

        result, gray = segment_one(image_path)

        out_row = row.to_dict()
        out_row.update(result)
        rows.append(out_row)

        save_overlay = False

        if result.get("segmentation_ok", False):
            if ok_saved < args.debug_overlays:
                save_overlay = True
                ok_saved += 1
        else:
            save_overlay = True

        if gray is not None and save_overlay:
            overlay = draw_overlay(gray, result, row["relative_path"])
            overlay_path = OVERLAY_DIR / safe_overlay_name(row["relative_path"])
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if result.get("segmentation_ok", False):
                ok_overlay_paths.append(overlay_path)

                if result.get("iris_confidence") == "high":
                    high_overlay_paths.append(overlay_path)
                elif result.get("iris_confidence") == "medium":
                    medium_overlay_paths.append(overlay_path)
            else:
                check_overlay_paths.append(overlay_path)

                if not result.get("pupil_found", False) or not result.get("iris_found", False):
                    fail_overlay_paths.append(overlay_path)

        if (len(rows)) % 250 == 0:
            print(f"Processate {len(rows)}/{len(df)} immagini")

    sdf = pd.DataFrame(rows)
    sdf.to_csv(REPORT_PATH, index=False)

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
    }

    if "iris_confidence" in sdf.columns:
        summary["iris_confidence_counts"] = sdf["iris_confidence"].fillna("none").value_counts().to_dict()

    for col in [
        "pupil_r",
        "pupil_score",
        "iris_r",
        "iris_score",
        "iris_pupil_radius_ratio",
        "iris_center_offset",
    ]:
        if col in sdf.columns:
            values = pd.to_numeric(sdf[col], errors="coerce").dropna()
            if len(values):
                summary[col] = {
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "mean": float(values.mean()),
                    "max": float(values.max()),
                }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    random.seed(42)

    def sample_paths(paths, k=25):
        if len(paths) <= k:
            return paths
        return random.sample(paths, k)

    save_contact_sheet(
        sample_paths(ok_overlay_paths, 25),
        "Segmentation V2 - OK examples",
        CONTACT_DIR / "segmentation_v2_ok_examples.png",
    )

    save_contact_sheet(
        sample_paths(check_overlay_paths, 25),
        "Segmentation V2 - CHECK examples",
        CONTACT_DIR / "segmentation_v2_check_examples.png",
    )

    save_contact_sheet(
        sample_paths(high_overlay_paths, 25),
        "Segmentation V2 - HIGH confidence OK examples",
        CONTACT_DIR / "segmentation_v2_high_confidence_examples.png",
    )

    save_contact_sheet(
        sample_paths(medium_overlay_paths, 25),
        "Segmentation V2 - MEDIUM confidence OK examples",
        CONTACT_DIR / "segmentation_v2_medium_confidence_examples.png",
    )

    save_contact_sheet(
        sample_paths(fail_overlay_paths, 25),
        "Segmentation V2 - FAIL examples",
        CONTACT_DIR / "segmentation_v2_fail_examples.png",
    )

    print()
    print("=== SEGMENTATION V2 SUMMARY ===")
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

    print(f"Report salvato in:     {REPORT_PATH}")
    print(f"Summary salvato in:    {SUMMARY_PATH}")
    print(f"Overlay salvati in:    {OVERLAY_DIR}")
    print(f"Contact sheet in:      {CONTACT_DIR}")


if __name__ == "__main__":
    main()