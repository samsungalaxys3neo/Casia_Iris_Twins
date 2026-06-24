from pathlib import Path
import math
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


METADATA_PATH = Path("iris_twins_project/data/metadata/metadata_final.csv")
FALLBACK_METADATA_PATH = Path("iris_twins_project/data/metadata/metadata_clean.csv")

OUTPUT_DIR = Path("iris_twins_project/data/segmentation")
OVERLAY_DIR = OUTPUT_DIR / "overlays"
CONTACT_DIR = OUTPUT_DIR / "contact_sheets"
REPORT_PATH = OUTPUT_DIR / "segmentation_report.csv"
SUMMARY_PATH = OUTPUT_DIR / "segmentation_summary.json"

N_DEBUG_OVERLAYS = 300
CONTACT_SHEET_COLS = 5
THUMB_SIZE = (180, 135)
LABEL_HEIGHT = 38


def load_metadata():
    path = METADATA_PATH if METADATA_PATH.exists() else FALLBACK_METADATA_PATH

    df = pd.read_csv(
        path,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    print(f"Uso metadata: {path}")
    return df


def find_pupil(gray):
    """
    Localizzazione semplice della pupilla:
    - la pupilla è una regione molto scura;
    - cerchiamo blob scuri abbastanza circolari;
    - scegliamo il candidato migliore.
    """

    h, w = gray.shape

    # Blur leggero per ridurre rumore
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold data-driven: prendiamo i pixel più scuri
    q = np.percentile(blur, 8)
    threshold = min(70, max(25, q + 5))

    dark = (blur < threshold).astype(np.uint8) * 255

    # Pulizia morfologica
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

        # Evitiamo blob troppo vicini ai bordi
        if x < 40 or x > w - 40 or y < 40 or y > h - 40:
            continue

        # Preferiamo regioni scure, circolari e ragionevolmente centrali
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

    # Interpola eventuali NaN
    valid = ~np.isnan(means)
    means[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), means[valid])

    # Smooth
    means_smooth = cv2.GaussianBlur(means.reshape(1, -1), (1, 9), 0).flatten()

    # Gradiente: passaggio iride -> sclera tende ad aumentare luminosità
    grad = np.gradient(means_smooth)

    best_idx = int(np.argmax(grad))
    best_r = radii[best_idx]
    best_grad = float(grad[best_idx])

    # Soglia molto permissiva: serve solo flaggare casi sospetti
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


def segment_image(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        return {
            "segmentation_ok": False,
            "pupil_found": False,
            "iris_found": False,
            "error": "cv2.imread failed",
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

    # Criteri preliminari di plausibilità
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

    sheet.save(output_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata()

    print("=== PRELIMINARY SEGMENTATION ===")
    print(f"Immagini da segmentare: {len(df)}")

    rows = []
    overlay_paths_ok = []
    overlay_paths_check = []

    for idx, row in df.iterrows():
        image_path = Path(row["absolute_path"])

        result, gray = segment_image(image_path)

        out_row = row.to_dict()
        out_row.update(result)
        rows.append(out_row)

        # Salviamo overlay solo per i primi N e per tutti i casi problematici
        save_debug = idx < N_DEBUG_OVERLAYS or not result.get("segmentation_ok", False)

        if gray is not None and save_debug:
            label = row["relative_path"]
            overlay = draw_overlay(gray, result, label)

            safe_name = row["relative_path"].replace("/", "__")
            overlay_path = OVERLAY_DIR / f"{safe_name}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if result.get("segmentation_ok", False):
                overlay_paths_ok.append(overlay_path)
            else:
                overlay_paths_check.append(overlay_path)

        if (idx + 1) % 250 == 0:
            print(f"Processate {idx + 1}/{len(df)} immagini")

    sdf = pd.DataFrame(rows)
    sdf.to_csv(REPORT_PATH, index=False)

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
    }

    if "iris_pupil_radius_ratio" in sdf.columns:
        ratio_values = sdf["iris_pupil_radius_ratio"].dropna()
        summary["iris_pupil_radius_ratio"] = {
            "min": float(ratio_values.min()) if len(ratio_values) else None,
            "median": float(ratio_values.median()) if len(ratio_values) else None,
            "mean": float(ratio_values.mean()) if len(ratio_values) else None,
            "max": float(ratio_values.max()) if len(ratio_values) else None,
        }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Contact sheet
    save_contact_sheet(
        overlay_paths_ok[:25],
        "Segmentation examples - OK",
        CONTACT_DIR / "segmentation_ok_examples.png",
    )

    save_contact_sheet(
        overlay_paths_check[:25],
        "Segmentation examples - CHECK",
        CONTACT_DIR / "segmentation_check_examples.png",
    )

    print()
    print("=== SEGMENTATION SUMMARY ===")
    print(f"Immagini totali:       {n}")
    print(f"Segmentation OK:       {n_ok}")
    print(f"Check/fail:            {n_fail}")
    print(f"OK rate:               {summary['segmentation_ok_rate']:.4f}")
    print(f"Pupil found:           {summary['pupil_found']}")
    print(f"Iris found:            {summary['iris_found']}")

    if "iris_pupil_radius_ratio" in summary:
        print()
        print("iris_pupil_radius_ratio:")
        for k, v in summary["iris_pupil_radius_ratio"].items():
            print(f"  {k}: {v}")

    print()
    print(f"Report salvato in:     {REPORT_PATH}")
    print(f"Overlay salvati in:    {OVERLAY_DIR}")
    print(f"Contact sheet in:      {CONTACT_DIR}")


if __name__ == "__main__":
    main()