from pathlib import Path
import argparse
import json
import math

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "daugman_strict_v3"
    / "normalized_daugman_strict_v3_report.csv"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "features_texture" / "daugman_strict_v3"

N_BANDS = 8
LBP_BINS = 256
LOG_SIGMAS = [1.5, 3.0, 5.0]


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


def project_relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_name(relative_path: str) -> str:
    return str(relative_path).replace("/", "__").replace("\\", "__")


def ensure_dirs(output_dir: Path):
    feature_dir = output_dir / "lbp_blob_features"
    preview_dir = output_dir / "previews"

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    return feature_dir, preview_dir


def load_grayscale(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def erode_mask(mask: np.ndarray) -> np.ndarray:
    """
    Coerente con la pipeline Gabor:
    erodiamo leggermente la maschera per evitare bordi sporchi.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    return eroded > 0


def normalize_valid_pixels(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalizza usando solo pixel validi.
    Le zone non valide vengono messe a 0.
    """
    img_f = img.astype(np.float32)
    valid = mask > 0

    if valid.sum() < 10:
        raise ValueError("Too few valid pixels")

    values = img_f[valid]
    mean = float(values.mean())
    std = float(values.std())

    if std < 1e-6:
        std = 1.0

    out = (img_f - mean) / std
    out[~valid] = 0.0
    return out.astype(np.float32)


def fill_invalid_with_median(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Per LBP vogliamo evitare che i pixel fuori maschera creino pattern falsi.
    Li riempiamo con la mediana dei pixel validi.
    """
    out = img.copy()
    valid = mask > 0

    if valid.sum() < 10:
        raise ValueError("Too few valid pixels")

    median_value = int(np.median(img[valid]))
    out[~valid] = median_value
    return out


def compute_lbp8(img: np.ndarray) -> np.ndarray:
    """
    LBP classico a 8 vicini, senza dipendenze esterne.
    Produce valori 0..255.
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    padded = np.pad(img, 1, mode="edge")
    center = padded[1:-1, 1:-1]

    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),
        (1, 1), (1, 0), (1, -1),
        (0, -1),
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)

    for bit, (dy, dx) in enumerate(offsets):
        neigh = padded[
            1 + dy:1 + dy + img.shape[0],
            1 + dx:1 + dx + img.shape[1],
        ]
        lbp |= ((neigh >= center).astype(np.uint8) << bit)

    return lbp


def band_slices(height: int, n_bands: int):
    """
    Divide la normalized iris in bande orizzontali.
    """
    edges = np.linspace(0, height, n_bands + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(n_bands)]


def extract_lbp_histograms(img: np.ndarray, mask: np.ndarray, n_bands=N_BANDS) -> np.ndarray:
    """
    Estrae istogrammi LBP per bande orizzontali.
    Output: n_bands * 256 valori.
    """
    valid_mask = erode_mask(mask)
    img_filled = fill_invalid_with_median(img, valid_mask)
    lbp = compute_lbp8(img_filled)

    features = []

    for y0, y1 in band_slices(img.shape[0], n_bands):
        band_lbp = lbp[y0:y1, :]
        band_mask = valid_mask[y0:y1, :]

        values = band_lbp[band_mask]

        if values.size == 0:
            hist = np.zeros(LBP_BINS, dtype=np.float32)
        else:
            hist, _ = np.histogram(values, bins=LBP_BINS, range=(0, 256))
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-8)

        features.append(hist)

    return np.concatenate(features).astype(np.float32)


def odd_kernel_size_for_sigma(sigma: float) -> int:
    k = int(2 * math.ceil(3 * sigma) + 1)
    return max(k, 3)


def extract_log_blob_features(norm_img: np.ndarray, mask: np.ndarray, n_bands=N_BANDS) -> np.ndarray:
    """
    Estrae feature Blob/LoG su più scale.

    Per ogni sigma e per ogni banda:
    - densità blob
    - risposta media LoG
    - risposta 95 percentile

    Output: len(LOG_SIGMAS) * n_bands * 3 valori.
    """
    valid_mask = erode_mask(mask)
    features = []

    for sigma in LOG_SIGMAS:
        ksize = odd_kernel_size_for_sigma(sigma)

        blurred = cv2.GaussianBlur(
            norm_img,
            ksize=(ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )

        log_response = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        response = (sigma ** 2) * np.abs(log_response)

        valid_values = response[valid_mask]

        if valid_values.size < 10:
            threshold = np.inf
        else:
            threshold = float(np.quantile(valid_values, 0.95))

        blob_map = (response >= threshold) & valid_mask

        for y0, y1 in band_slices(norm_img.shape[0], n_bands):
            band_mask = valid_mask[y0:y1, :]
            band_response = response[y0:y1, :]
            band_blob = blob_map[y0:y1, :]

            if band_mask.sum() == 0:
                density = 0.0
                mean_response = 0.0
                q95_response = 0.0
            else:
                values = band_response[band_mask]
                density = float(band_blob[band_mask].mean())
                mean_response = float(values.mean())
                q95_response = float(np.quantile(values, 0.95))

            features.extend([density, mean_response, q95_response])

    return np.array(features, dtype=np.float32)


def make_lbp_preview(lbp_feature: np.ndarray, output_path: Path):
    """
    Preview semplice degli istogrammi LBP.
    Non è usata per matching, serve solo per controllo visivo.
    """
    mat = lbp_feature.reshape(N_BANDS, LBP_BINS)
    mat = mat / (mat.max() + 1e-8)
    preview = (mat * 255).astype(np.uint8)
    preview = cv2.resize(preview, (512, 160), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_path), preview)


def save_feature_npz(path: Path, lbp: np.ndarray, blob: np.ndarray, relative_path: str):
    combined = np.concatenate([lbp, blob]).astype(np.float32)

    np.savez_compressed(
        path,
        lbp=lbp.astype(np.float32),
        blob=blob.astype(np.float32),
        combined=combined,
        relative_path=np.array(relative_path),
        n_bands=np.array(N_BANDS),
        log_sigmas=np.array(LOG_SIGMAS, dtype=np.float32),
    )


def run_texture_feature_extraction(input_report: Path, output_dir: Path):
    input_report = resolve_from_project(input_report)
    output_dir = resolve_from_project(output_dir)

    if not input_report.exists():
        raise FileNotFoundError(f"Input report non trovato: {input_report}")

    feature_dir, preview_dir = ensure_dirs(output_dir)

    feature_report = output_dir / "texture_features_report.csv"
    summary_path = output_dir / "texture_features_summary.json"

    dtype_map = {
        "family_id": str,
        "twin_id": str,
        "eye": str,
        "subject_id": str,
        "iris_id": str,
        "image_idx": str,
        "filename": str,
        "relative_path": str,
        "normalized_path": str,
        "mask_path": str,
        "masked_preview_path": str,
    }

    df = pd.read_csv(input_report, dtype=dtype_map)

    if "normalization_ok" in df.columns:
        df = df[as_bool(df["normalization_ok"])].copy()

    df = df.reset_index(drop=True)

    print("=== LBP + BLOB FEATURE EXTRACTION ===")
    print(f"Input report: {project_display_path(input_report)}")
    print(f"Input images: {len(df)}")
    print(f"Output dir: {project_display_path(output_dir)}")
    print()

    rows = []

    for idx, row in df.iterrows():
        out_row = row.to_dict()

        try:
            norm_path = resolve_from_project(Path(str(row["normalized_path"])))
            mask_path = resolve_from_project(Path(str(row["mask_path"])))

            out_row["normalized_path"] = project_relative_or_absolute(norm_path)
            out_row["mask_path"] = project_relative_or_absolute(mask_path)

            img = load_grayscale(norm_path)
            mask = load_grayscale(mask_path)

            valid_mask = erode_mask(mask)
            valid_ratio = float(valid_mask.mean())

            norm_img = normalize_valid_pixels(img, valid_mask)

            lbp_feature = extract_lbp_histograms(img, valid_mask)
            blob_feature = extract_log_blob_features(norm_img, valid_mask)

            base = safe_name(row["relative_path"])

            feature_path = feature_dir / f"{base}_lbp_blob.npz"
            preview_path = preview_dir / f"{base}_lbp_preview.png"

            save_feature_npz(feature_path, lbp_feature, blob_feature, row["relative_path"])
            make_lbp_preview(lbp_feature, preview_path)

            out_row["texture_feature_ok"] = True
            out_row["texture_feature_error"] = ""
            out_row["lbp_blob_path"] = project_relative_or_absolute(feature_path)
            out_row["lbp_preview_path"] = project_relative_or_absolute(preview_path)
            out_row["lbp_dim"] = int(lbp_feature.shape[0])
            out_row["blob_dim"] = int(blob_feature.shape[0])
            out_row["combined_dim"] = int(lbp_feature.shape[0] + blob_feature.shape[0])
            out_row["texture_valid_ratio"] = valid_ratio
            out_row["blob_mean"] = float(blob_feature.mean())
            out_row["blob_std"] = float(blob_feature.std())

        except Exception as e:
            out_row["texture_feature_ok"] = False
            out_row["texture_feature_error"] = str(e)
            out_row["lbp_blob_path"] = ""
            out_row["lbp_preview_path"] = ""
            out_row["lbp_dim"] = 0
            out_row["blob_dim"] = 0
            out_row["combined_dim"] = 0
            out_row["texture_valid_ratio"] = 0.0
            out_row["blob_mean"] = 0.0
            out_row["blob_std"] = 0.0

        rows.append(out_row)

        if (idx + 1) % 250 == 0:
            print(f"Processed {idx + 1}/{len(df)}")

    fdf = pd.DataFrame(rows)

    old_abs_col = "absolute" + "_path"
    if old_abs_col in fdf.columns:
        fdf = fdf.drop(columns=[old_abs_col])

    fdf.to_csv(feature_report, index=False)

    ok_df = fdf[as_bool(fdf["texture_feature_ok"])].copy()

    summary = {
        "input_report": project_display_path(input_report),
        "output_dir": project_display_path(output_dir),
        "input_images": int(len(df)),
        "texture_feature_ok": int(len(ok_df)),
        "texture_feature_failed": int(len(df) - len(ok_df)),
        "n_bands": N_BANDS,
        "lbp_bins": LBP_BINS,
        "log_sigmas": LOG_SIGMAS,
        "output_feature_report": project_display_path(feature_report),
        "feature_dir": project_display_path(feature_dir),
        "preview_dir": project_display_path(preview_dir),
    }

    if len(ok_df) > 0:
        for col in ["texture_valid_ratio", "blob_mean", "blob_std"]:
            values = pd.to_numeric(ok_df[col], errors="coerce").dropna()
            summary[col] = {
                "min": float(values.min()),
                "q1": float(values.quantile(0.25)),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "q3": float(values.quantile(0.75)),
                "max": float(values.max()),
            }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== SUMMARY ===")
    print(f"Input images:   {summary['input_images']}")
    print(f"Feature OK:     {summary['texture_feature_ok']}")
    print(f"Feature failed: {summary['texture_feature_failed']}")
    print(f"Feature report: {project_display_path(feature_report)}")
    print(f"Summary:        {project_display_path(summary_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-report",
        type=Path,
        default=DEFAULT_INPUT_REPORT,
        help="Path al normalized_daugman_strict_v3_report.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella output per feature LBP + Blob.",
    )

    args = parser.parse_args()

    run_texture_feature_extraction(
        input_report=args.input_report,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()