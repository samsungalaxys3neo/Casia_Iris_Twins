from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEXTURE_REPORT = (
    PROJECT_ROOT
    / "data"
    / "features_texture"
    / "daugman_strict_v3"
    / "texture_features_report.csv"
)

DEFAULT_INPUT_PAIRS = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "daugman_strict_v3"
    / "pairs_features.csv.gz"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "texture_similarity" / "daugman_strict_v3"


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


def chi_square_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distanza Chi-square per istogrammi LBP.
    Più bassa = texture più simile.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(0.5 * np.sum(((a - b) ** 2) / (a + b + 1e-8)))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean normalizzata per lunghezza vettore.
    """
    if a.size == 0:
        return float("nan")
    return float(np.linalg.norm(a - b) / math.sqrt(a.size))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    1 - cosine similarity.
    Più bassa = più simile.
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 1.0

    sim = float(np.dot(a, b) / denom)
    return 1.0 - sim


def load_feature_npz(path: Path):
    data = np.load(path)
    return {
        "lbp": data["lbp"].astype(np.float32),
        "blob": data["blob"].astype(np.float32),
        "combined": data["combined"].astype(np.float32),
    }


def standardize_matrix(mat: np.ndarray):
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def load_all_features(texture_df: pd.DataFrame):
    """
    Carica tutte le feature in RAM.
    Con 1465 immagini è leggero.
    """
    features = {}
    blob_vectors = []

    for _, row in texture_df.iterrows():
        rel = str(row["relative_path"])
        path = resolve_from_project(Path(str(row["lbp_blob_path"])))

        item = load_feature_npz(path)
        features[rel] = item
        blob_vectors.append(item["blob"])

    if len(blob_vectors) == 0:
        raise ValueError("No texture features available.")

    blob_matrix = np.vstack(blob_vectors).astype(np.float32)
    blob_mean, blob_std = standardize_matrix(blob_matrix)

    for _, item in features.items():
        item["blob_std"] = (item["blob"] - blob_mean) / blob_std

    return features


def summarize_by_relation(df: pd.DataFrame, metric: str):
    rows = []

    for relation, group in df.groupby("relation"):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()

        if len(values) == 0:
            continue

        rows.append(
            {
                "metric": metric,
                "relation": relation,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return rows


def run_lbp_blob_matching(texture_report: Path, input_pairs: Path, output_dir: Path):
    texture_report = resolve_from_project(texture_report)
    input_pairs = resolve_from_project(input_pairs)
    output_dir = resolve_from_project(output_dir)

    if not texture_report.exists():
        raise FileNotFoundError(f"Texture report non trovato: {texture_report}")

    if not input_pairs.exists():
        raise FileNotFoundError(f"Input pairs non trovato: {input_pairs}")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_distances = output_dir / "lbp_blob_distances.csv.gz"
    output_stats = output_dir / "lbp_blob_relation_stats.csv"
    summary_path = output_dir / "lbp_blob_matching_summary.json"

    texture_df = pd.read_csv(
        texture_report,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "subject_id": str,
            "iris_id": str,
            "image_idx": str,
            "filename": str,
            "relative_path": str,
            "lbp_blob_path": str,
            "lbp_preview_path": str,
        },
    )

    if "texture_feature_ok" in texture_df.columns:
        texture_df = texture_df[as_bool(texture_df["texture_feature_ok"])].copy()

    print("=== LBP + BLOB MATCHING ===")
    print(f"Texture report:      {project_display_path(texture_report)}")
    print(f"Texture features OK: {len(texture_df)}")
    print(f"Input pairs:         {project_display_path(input_pairs)}")
    print(f"Output dir:          {project_display_path(output_dir)}")
    print()

    features = load_all_features(texture_df)
    available = set(features.keys())

    pairs_df = pd.read_csv(
        input_pairs,
        compression="gzip",
        dtype={
            "family_id_1": str,
            "family_id_2": str,
            "subject_id_1": str,
            "subject_id_2": str,
            "iris_id_1": str,
            "iris_id_2": str,
            "twin_id_1": str,
            "twin_id_2": str,
            "eye_1": str,
            "eye_2": str,
            "image_1": str,
            "image_2": str,
            "relation": str,
        },
    )

    rows = []
    failed = 0

    for idx, row in pairs_df.iterrows():
        image_1 = str(row["image_1"])
        image_2 = str(row["image_2"])

        if image_1 not in available or image_2 not in available:
            failed += 1
            continue

        f1 = features[image_1]
        f2 = features[image_2]

        lbp_chi2 = chi_square_distance(f1["lbp"], f2["lbp"])
        blob_euclid = euclidean_distance(f1["blob_std"], f2["blob_std"])
        blob_cosine = cosine_distance(f1["blob_std"], f2["blob_std"])

        out = row.to_dict()
        out["lbp_chi2_distance"] = lbp_chi2
        out["blob_euclidean_distance"] = blob_euclid
        out["blob_cosine_distance"] = blob_cosine

        old_abs_col = "absolute" + "_path"
        if old_abs_col in out:
            out.pop(old_abs_col)

        rows.append(out)

        if (idx + 1) % 10000 == 0:
            print(f"Processed pairs: {idx + 1}/{len(pairs_df)}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_distances, index=False, compression="gzip")

    stats_rows = []
    for metric in [
        "lbp_chi2_distance",
        "blob_euclidean_distance",
        "blob_cosine_distance",
    ]:
        stats_rows.extend(summarize_by_relation(out_df, metric))

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_stats, index=False)

    summary = {
        "texture_report": project_display_path(texture_report),
        "input_pairs_path": project_display_path(input_pairs),
        "output_dir": project_display_path(output_dir),
        "texture_features_ok": int(len(texture_df)),
        "input_pairs": int(len(pairs_df)),
        "successful_pairs": int(len(out_df)),
        "failed_pairs": int(failed),
        "output_distances": project_display_path(output_distances),
        "output_stats": project_display_path(output_stats),
        "main_interpretation": (
            "Lower distance means higher texture similarity. "
            "LBP uses Chi-square distance on band-wise histograms. "
            "Blob/LoG uses Euclidean and cosine distances on standardized blob descriptors."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== SUMMARY ===")
    print(f"Input pairs:      {summary['input_pairs']}")
    print(f"Successful pairs: {summary['successful_pairs']}")
    print(f"Failed pairs:     {summary['failed_pairs']}")
    print(f"Distances:        {project_display_path(output_distances)}")
    print(f"Stats:            {project_display_path(output_stats)}")
    print(f"Summary:          {project_display_path(summary_path)}")

    print()
    print("=== QUICK RELATION STATS ===")
    print(stats_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--texture-report",
        type=Path,
        default=DEFAULT_TEXTURE_REPORT,
        help="Path al texture_features_report.csv.",
    )
    parser.add_argument(
        "--input-pairs",
        type=Path,
        default=DEFAULT_INPUT_PAIRS,
        help="Path a pairs_features.csv.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella output per similarity LBP + Blob.",
    )

    args = parser.parse_args()

    run_lbp_blob_matching(
        texture_report=args.texture_report,
        input_pairs=args.input_pairs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()