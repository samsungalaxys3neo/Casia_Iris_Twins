from pathlib import Path
import json
import math

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_REPORT = PROJECT_ROOT / "data/normalized/daugman_strict_v3/normalized_daugman_strict_v3_report.csv"

OUTPUT_DIR = PROJECT_ROOT / "data/features/daugman_strict_v3"
CODES_DIR = OUTPUT_DIR / "gabor_codes"
PREVIEW_DIR = OUTPUT_DIR / "code_previews"
CONTACT_DIR = OUTPUT_DIR / "contact_sheets"

FEATURE_REPORT = OUTPUT_DIR / "features_report.csv"
SUMMARY_PATH = OUTPUT_DIR / "features_summary.json"

THUMB_SIZE = (256, 64)
LABEL_HEIGHT = 38
CONTACT_COLS = 3


# Filtri Gabor-style.
# Usiamo due orientazioni e due scale.
# Non è il codice proprietario di Daugman, ma una baseline sperimentale Gabor-based.
GABOR_FILTERS = [
    {"name": "theta0_lambda8", "theta": 0.0, "lambda": 8.0, "sigma": 4.0, "gamma": 0.5},
    {"name": "theta90_lambda8", "theta": np.pi / 2, "lambda": 8.0, "sigma": 4.0, "gamma": 0.5},
    {"name": "theta0_lambda16", "theta": 0.0, "lambda": 16.0, "sigma": 6.0, "gamma": 0.5},
    {"name": "theta90_lambda16", "theta": np.pi / 2, "lambda": 16.0, "sigma": 6.0, "gamma": 0.5},
]


def safe_name(relative_path: str) -> str:
    return relative_path.replace("/", "__").replace("\\", "__")


def make_gabor_kernel(filter_cfg, ksize=21):
    kernel = cv2.getGaborKernel(
        ksize=(ksize, ksize),
        sigma=filter_cfg["sigma"],
        theta=filter_cfg["theta"],
        lambd=filter_cfg["lambda"],
        gamma=filter_cfg["gamma"],
        psi=0,
        ktype=cv2.CV_32F,
    )

    # Rimuove DC component, utile per evitare bias di luminosità.
    kernel = kernel - kernel.mean()

    norm = np.sqrt((kernel ** 2).sum())
    if norm > 0:
        kernel = kernel / norm

    return kernel.astype(np.float32)


def normalize_valid_pixels(img, mask):
    """
    Normalizza l'immagine usando solo i pixel validi.
    Le zone non valide vengono poste a 0 dopo la normalizzazione.
    """
    img_f = img.astype(np.float32)
    valid = mask > 0

    if valid.sum() < 10:
        return None

    values = img_f[valid]
    mean = float(values.mean())
    std = float(values.std())

    if std < 1e-6:
        std = 1.0

    norm = (img_f - mean) / std
    norm[~valid] = 0.0

    return norm


def erode_mask(mask):
    """
    Erode leggermente la maschera per evitare che i filtri Gabor usino
    pixel troppo vicini ai bordi non validi.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    return eroded > 0


def extract_code(norm_img, mask):
    """
    Applica i filtri Gabor e genera codice binario.
    Per ogni filtro:
    - response = filter2D(norm_img)
    - bit = response > 0
    Il mask viene replicato su tutti i filtri.
    """
    valid_mask = erode_mask(mask)

    code_bits = []
    mask_bits = []

    for cfg in GABOR_FILTERS:
        kernel = make_gabor_kernel(cfg)
        response = cv2.filter2D(norm_img, cv2.CV_32F, kernel)

        bits = response > 0

        code_bits.append(bits)
        mask_bits.append(valid_mask)

    code = np.stack(code_bits, axis=0).astype(np.uint8)
    code_mask = np.stack(mask_bits, axis=0).astype(np.uint8)

    return code, code_mask


def save_code_npz(code_path, code, code_mask, relative_path):
    """
    Salva codice e maschera.
    Non facciamo packbits per ora: il file compresso .npz è più semplice da leggere
    nel prossimo step di matching.
    """
    filter_names = np.array([cfg["name"] for cfg in GABOR_FILTERS])

    np.savez_compressed(
        code_path,
        code=code,
        mask=code_mask,
        filter_names=filter_names,
        relative_path=np.array(relative_path),
    )


def make_code_preview(code, code_mask, output_path):
    """
    Crea preview del primo filtro:
    - bianco = bit 1
    - nero = bit 0
    - grigio = pixel non valido
    """
    first_code = code[0]
    first_mask = code_mask[0] > 0

    preview = np.full(first_code.shape, 128, dtype=np.uint8)
    preview[first_mask & (first_code == 1)] = 255
    preview[first_mask & (first_code == 0)] = 0

    cv2.imwrite(str(output_path), preview)


def make_contact_sheet(df, title, output_path, n=45):
    if len(df) == 0:
        return

    if len(df) > n:
        df = df.sample(n=n, random_state=42).copy()

    font = ImageFont.load_default()
    thumbs = []

    for _, row in df.iterrows():
        path = Path(row["code_preview_path"])

        if not path.exists():
            continue

        img = Image.open(path).convert("L")
        img = img.resize(THUMB_SIZE)

        canvas = Image.new("L", (THUMB_SIZE[0], THUMB_SIZE[1] + LABEL_HEIGHT), color=255)
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        label1 = str(row["relative_path"])[:42]
        label2 = f"valid_bits={row['valid_bit_ratio']:.3f}"

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
    CODES_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_REPORT,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    # Usiamo solo normalizzazioni OK.
    df = df[df["normalization_ok"] == True].copy().reset_index(drop=True)

    print("=== GABOR-STYLE FEATURE EXTRACTION ===")
    print(f"Input report: {INPUT_REPORT}")
    print(f"Immagini normalizzate OK: {len(df)}")
    print(f"Numero filtri Gabor: {len(GABOR_FILTERS)}")
    print()

    rows = []

    for idx, row in df.iterrows():
        norm_path = Path(row["normalized_path"])
        mask_path = Path(row["mask_path"])

        out_row = row.to_dict()

        if not norm_path.exists():
            out_row["feature_ok"] = False
            out_row["feature_error"] = f"normalized image not found: {norm_path}"
            rows.append(out_row)
            continue

        if not mask_path.exists():
            out_row["feature_ok"] = False
            out_row["feature_error"] = f"mask not found: {mask_path}"
            rows.append(out_row)
            continue

        img = cv2.imread(str(norm_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            out_row["feature_ok"] = False
            out_row["feature_error"] = "cv2.imread normalized failed"
            rows.append(out_row)
            continue

        if mask is None:
            out_row["feature_ok"] = False
            out_row["feature_error"] = "cv2.imread mask failed"
            rows.append(out_row)
            continue

        try:
            norm_img = normalize_valid_pixels(img, mask)

            if norm_img is None:
                out_row["feature_ok"] = False
                out_row["feature_error"] = "too few valid pixels"
                rows.append(out_row)
                continue

            code, code_mask = extract_code(norm_img, mask)

            base = safe_name(row["relative_path"])

            code_path = CODES_DIR / f"{base}.npz"
            preview_path = PREVIEW_DIR / f"{base}_code_preview.png"

            save_code_npz(code_path, code, code_mask, row["relative_path"])
            make_code_preview(code, code_mask, preview_path)

            valid_bit_ratio = float(code_mask.mean())
            bit_one_ratio = float(code[code_mask > 0].mean()) if code_mask.sum() > 0 else 0.0

            out_row["feature_ok"] = True
            out_row["feature_error"] = ""
            out_row["code_path"] = str(code_path)
            out_row["code_preview_path"] = str(preview_path)
            out_row["num_filters"] = len(GABOR_FILTERS)
            out_row["code_height"] = int(code.shape[1])
            out_row["code_width"] = int(code.shape[2])
            out_row["valid_bit_ratio"] = valid_bit_ratio
            out_row["bit_one_ratio"] = bit_one_ratio

        except Exception as e:
            out_row["feature_ok"] = False
            out_row["feature_error"] = str(e)

        rows.append(out_row)

        if (idx + 1) % 250 == 0:
            print(f"Estratte feature {idx + 1}/{len(df)} immagini")

    fdf = pd.DataFrame(rows)
    fdf.to_csv(FEATURE_REPORT, index=False)

    ok_df = fdf[fdf["feature_ok"] == True].copy()

    summary = {
        "input_images": int(len(df)),
        "feature_ok": int(len(ok_df)),
        "feature_failed": int(len(df) - len(ok_df)),
        "num_filters": len(GABOR_FILTERS),
        "filters": GABOR_FILTERS,
    }

    if len(ok_df) > 0:
        for col in ["valid_bit_ratio", "bit_one_ratio"]:
            values = pd.to_numeric(ok_df[col], errors="coerce").dropna()
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
        title="Gabor-style code previews",
        output_path=CONTACT_DIR / "gabor_code_previews.png",
        n=45,
    )

    low_valid = ok_df.sort_values("valid_bit_ratio", ascending=True).head(45)
    make_contact_sheet(
        low_valid,
        title="Lowest valid bit ratio code previews",
        output_path=CONTACT_DIR / "lowest_valid_bit_ratio_previews.png",
        n=45,
    )

    print()
    print("=== FEATURE EXTRACTION SUMMARY ===")
    print(f"Input immagini:   {summary['input_images']}")
    print(f"Feature OK:       {summary['feature_ok']}")
    print(f"Feature failed:   {summary['feature_failed']}")
    print(f"Numero filtri:    {summary['num_filters']}")

    print()
    for col in ["valid_bit_ratio", "bit_one_ratio"]:
        if col in summary:
            print(col)
            for k, v in summary[col].items():
                print(f"  {k}: {v}")
            print()

    print(f"Feature report:   {FEATURE_REPORT}")
    print(f"Summary:          {SUMMARY_PATH}")
    print(f"Codici in:        {CODES_DIR}")
    print(f"Preview in:       {PREVIEW_DIR}")
    print(f"Contact sheets:   {CONTACT_DIR}")


if __name__ == "__main__":
    main()