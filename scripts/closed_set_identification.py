from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "daugman_strict_v3" / "features_report.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "identification_closed_set" / "daugman_strict_v3"


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


def read_features(input_features: Path) -> pd.DataFrame:
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
        "normalized_path": str,
        "mask_path": str,
        "masked_preview_path": str,
        "code_path": str,
        "code_preview_path": str,
    }

    df = pd.read_csv(input_features, dtype=dtype_map)

    df["feature_ok"] = (
        df["feature_ok"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    df["valid_bit_ratio"] = pd.to_numeric(df["valid_bit_ratio"], errors="coerce")

    return df


def load_code_npz(path: Path):
    data = np.load(str(path))
    code = data["code"].astype(bool)
    mask = data["mask"].astype(bool)
    return code, mask


def masked_hamming_distance(code1, mask1, code2, mask2, shifts, min_common_bits=1000):
    best_hd = None
    best_shift = None
    best_common_bits = 0

    for shift in shifts:
        shifted_code2 = np.roll(code2, shift=shift, axis=2)
        shifted_mask2 = np.roll(mask2, shift=shift, axis=2)

        common_mask = mask1 & shifted_mask2
        common_bits = int(common_mask.sum())

        if common_bits < min_common_bits:
            continue

        xor = np.logical_xor(code1, shifted_code2)
        diff_bits = int((xor & common_mask).sum())

        hd = diff_bits / common_bits

        if best_hd is None or hd < best_hd:
            best_hd = float(hd)
            best_shift = int(shift)
            best_common_bits = int(common_bits)

    if best_hd is None:
        return {
            "match_ok": False,
            "score": np.inf,
            "best_shift": np.nan,
            "common_bits": best_common_bits,
        }

    return {
        "match_ok": True,
        "score": best_hd,
        "best_shift": best_shift,
        "common_bits": best_common_bits,
    }


def relation_to_probe(probe, gallery):
    same_family = str(probe["family_id"]) == str(gallery["family_id"])
    same_subject = str(probe["subject_id"]) == str(gallery["subject_id"])
    same_iris = str(probe["iris_id"]) == str(gallery["iris_id"])
    same_eye = str(probe["eye"]) == str(gallery["eye"])

    if same_iris:
        return "genuine_gallery"

    if same_subject and not same_eye:
        return "same_subject_different_eye"

    if same_family and not same_subject and same_eye:
        return "twin_same_eye"

    if same_family and not same_subject and not same_eye:
        return "twin_cross_eye"

    return "unrelated"


def choose_gallery_probe(df, selection="best_valid", seed=42):
    gallery_rows = []
    probe_rows = []

    rng = np.random.default_rng(seed)

    for iris_id, group in df.groupby("iris_id"):
        group = group.copy()

        if selection == "best_valid":
            group = group.sort_values(
                ["valid_bit_ratio", "relative_path"],
                ascending=[False, True],
            )
            gallery = group.iloc[0]

        elif selection == "first":
            group = group.sort_values("relative_path")
            gallery = group.iloc[0]

        elif selection == "random":
            gallery = group.iloc[int(rng.integers(0, len(group)))]

        else:
            raise ValueError(f"Unknown gallery selection: {selection}")

        gallery_rows.append(gallery.to_dict())

        probes = group[group.index != gallery.name]
        for _, probe in probes.iterrows():
            probe_rows.append(probe.to_dict())

    return gallery_rows, probe_rows


def compute_stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return None

    return {
        "n": int(len(arr)),
        "min": float(np.min(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q3": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def make_cmc(results_df, max_rank):
    rows = []

    for k in range(1, max_rank + 1):
        cms = float((results_df["true_rank"] <= k).mean())
        rows.append(
            {
                "rank": int(k),
                "CMS": cms,
            }
        )

    return pd.DataFrame(rows)


def plot_cmc(cmc_df, plots_dir: Path):
    plt.figure()
    plt.plot(cmc_df["rank"], cmc_df["CMS"], marker="o")
    plt.xlabel("Rank k")
    plt.ylabel("CMS(k)")
    plt.title("Closed-set CMC curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "cmc_curve.png", dpi=160)
    plt.close()


def plot_rank_histogram(results_df, plots_dir: Path):
    ranks = results_df["true_rank"].dropna().astype(int)
    clipped = ranks.clip(upper=20)

    plt.figure()
    plt.hist(clipped, bins=np.arange(1, 22) - 0.5)
    plt.xlabel("True rank, clipped at 20")
    plt.ylabel("Number of probes")
    plt.title("Distribution of true identity rank")
    plt.tight_layout()
    plt.savefig(plots_dir / "true_rank_histogram.png", dpi=160)
    plt.close()


def plot_margin_histogram(results_df, plots_dir: Path):
    values = results_df["margin_best_impostor_minus_true"].dropna()

    plt.figure()
    plt.hist(values, bins=60)
    plt.xlabel("Best impostor score - true score")
    plt.ylabel("Number of probes")
    plt.title("Identification margin")
    plt.tight_layout()
    plt.savefig(plots_dir / "identification_margin_histogram.png", dpi=160)
    plt.close()


def pct(count, total):
    return float(count / total) if total else 0.0


def run_closed_set(
    input_features: Path,
    output_dir: Path,
    max_shift: int,
    shift_step: int,
    min_common_bits: int,
    gallery_selection: str,
    seed: int,
    top_k: int,
    cmc_max_rank: int,
):
    if not input_features.exists():
        raise FileNotFoundError(f"Feature report non trovato: {input_features}")

    plots_dir = output_dir / "plots"

    results_csv = output_dir / "closed_set_results.csv"
    topk_csv_gz = output_dir / "closed_set_topk.csv.gz"
    cmc_csv = output_dir / "cmc_curve.csv"
    summary_json = output_dir / "closed_set_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    shifts = list(range(-max_shift, max_shift + 1, shift_step))

    print("=== CLOSED-SET IDENTIFICATION 1:N ===")
    print(f"Input features:       {project_display_path(input_features)}")
    print(f"Output dir:           {project_display_path(output_dir)}")
    print(f"Gallery selection:    {gallery_selection}")
    print(f"Shifts:               {shifts}")
    print(f"Min common bits:      {min_common_bits}")
    print()

    df = read_features(input_features)
    df = df[df["feature_ok"] == True].copy().reset_index(drop=True)

    gallery_rows, probe_rows = choose_gallery_probe(
        df,
        selection=gallery_selection,
        seed=seed,
    )

    print(f"Feature images OK:    {len(df)}")
    print(f"Gallery templates:    {len(gallery_rows)}")
    print(f"Probe templates:      {len(probe_rows)}")
    print()

    unique_paths = sorted(
        set([str(r["code_path"]) for r in gallery_rows])
        | set([str(r["code_path"]) for r in probe_rows])
    )

    print(f"Codici unici da caricare: {len(unique_paths)}")

    code_cache = {}

    for idx, path_value in enumerate(unique_paths):
        resolved_path = resolve_from_project(Path(path_value))

        if not resolved_path.exists():
            raise FileNotFoundError(f"Codice Gabor non trovato: {resolved_path}")

        code_cache[path_value] = load_code_npz(resolved_path)

        if (idx + 1) % 250 == 0:
            print(f"Caricati {idx + 1}/{len(unique_paths)} codici")

    results = []
    topk_rows = []

    total_comparisons = len(probe_rows) * len(gallery_rows)
    failed_comparisons = 0
    comparison_counter = 0

    for p_idx, probe in enumerate(probe_rows):
        probe_code, probe_mask = code_cache[str(probe["code_path"])]

        ranking = []

        for gallery in gallery_rows:
            gallery_code, gallery_mask = code_cache[str(gallery["code_path"])]

            match = masked_hamming_distance(
                probe_code,
                probe_mask,
                gallery_code,
                gallery_mask,
                shifts=shifts,
                min_common_bits=min_common_bits,
            )

            if not match["match_ok"]:
                failed_comparisons += 1

            rel = relation_to_probe(probe, gallery)

            item = {
                "gallery_relative_path": gallery["relative_path"],
                "gallery_code_path": gallery["code_path"],
                "gallery_family_id": gallery["family_id"],
                "gallery_subject_id": gallery["subject_id"],
                "gallery_iris_id": gallery["iris_id"],
                "gallery_eye": gallery["eye"],
                "score": match["score"],
                "match_ok": match["match_ok"],
                "best_shift": match["best_shift"],
                "common_bits": match["common_bits"],
                "relation_to_probe": rel,
                "is_true": rel == "genuine_gallery",
                "is_twin": rel in {"twin_same_eye", "twin_cross_eye"},
                "is_twin_same_eye": rel == "twin_same_eye",
                "is_twin_cross_eye": rel == "twin_cross_eye",
            }

            ranking.append(item)
            comparison_counter += 1

        ranking = sorted(
            ranking,
            key=lambda x: (x["score"], str(x["gallery_iris_id"])),
        )

        for rank_idx, item in enumerate(ranking, start=1):
            item["rank"] = rank_idx

        true_items = [r for r in ranking if r["is_true"]]

        if not true_items:
            raise RuntimeError(
                f"Nessun template gallery genuine trovato per probe {probe['relative_path']}"
            )

        true_item = true_items[0]
        best_item = ranking[0]

        impostor_items = [r for r in ranking if not r["is_true"]]

        if not impostor_items:
            raise RuntimeError("La gallery contiene solo il genuine, impossibile calcolare best impostor.")

        best_impostor = impostor_items[0]

        twin_items = [r for r in ranking if r["is_twin"]]
        twin_same_eye_items = [r for r in ranking if r["is_twin_same_eye"]]
        twin_cross_eye_items = [r for r in ranking if r["is_twin_cross_eye"]]

        best_twin = twin_items[0] if len(twin_items) > 0 else None
        best_twin_same_eye = twin_same_eye_items[0] if len(twin_same_eye_items) > 0 else None
        best_twin_cross_eye = twin_cross_eye_items[0] if len(twin_cross_eye_items) > 0 else None

        true_score = float(true_item["score"])
        best_impostor_score = float(best_impostor["score"])

        result = {
            "probe_relative_path": probe["relative_path"],
            "probe_code_path": probe["code_path"],
            "probe_family_id": probe["family_id"],
            "probe_subject_id": probe["subject_id"],
            "probe_iris_id": probe["iris_id"],
            "probe_eye": probe["eye"],

            "true_rank": int(true_item["rank"]),
            "rank1_correct": bool(best_item["is_true"]),
            "true_score": true_score,
            "rank1_score": float(best_item["score"]),
            "best_impostor_score": best_impostor_score,
            "margin_best_impostor_minus_true": best_impostor_score - true_score,

            "rank1_gallery_iris_id": best_item["gallery_iris_id"],
            "rank1_gallery_subject_id": best_item["gallery_subject_id"],
            "rank1_gallery_family_id": best_item["gallery_family_id"],
            "rank1_gallery_eye": best_item["gallery_eye"],
            "rank1_relation_to_probe": best_item["relation_to_probe"],
            "rank1_is_twin": bool(best_item["is_twin"]),
            "rank1_is_twin_same_eye": bool(best_item["is_twin_same_eye"]),
            "rank1_is_twin_cross_eye": bool(best_item["is_twin_cross_eye"]),

            "best_impostor_iris_id": best_impostor["gallery_iris_id"],
            "best_impostor_subject_id": best_impostor["gallery_subject_id"],
            "best_impostor_family_id": best_impostor["gallery_family_id"],
            "best_impostor_eye": best_impostor["gallery_eye"],
            "best_impostor_relation_to_probe": best_impostor["relation_to_probe"],
            "best_impostor_is_twin": bool(best_impostor["is_twin"]),
            "best_impostor_is_twin_same_eye": bool(best_impostor["is_twin_same_eye"]),
            "best_impostor_is_twin_cross_eye": bool(best_impostor["is_twin_cross_eye"]),

            "min_twin_rank": int(best_twin["rank"]) if best_twin is not None else np.nan,
            "best_twin_score": float(best_twin["score"]) if best_twin is not None else np.nan,
            "margin_best_twin_minus_true": float(best_twin["score"] - true_score) if best_twin is not None else np.nan,

            "min_twin_same_eye_rank": int(best_twin_same_eye["rank"]) if best_twin_same_eye is not None else np.nan,
            "best_twin_same_eye_score": float(best_twin_same_eye["score"]) if best_twin_same_eye is not None else np.nan,

            "min_twin_cross_eye_rank": int(best_twin_cross_eye["rank"]) if best_twin_cross_eye is not None else np.nan,
            "best_twin_cross_eye_score": float(best_twin_cross_eye["score"]) if best_twin_cross_eye is not None else np.nan,

            "true_common_bits": int(true_item["common_bits"]),
            "true_best_shift": true_item["best_shift"],
        }

        results.append(result)

        for item in ranking[:top_k]:
            topk_rows.append(
                {
                    "probe_relative_path": probe["relative_path"],
                    "probe_family_id": probe["family_id"],
                    "probe_subject_id": probe["subject_id"],
                    "probe_iris_id": probe["iris_id"],
                    "probe_eye": probe["eye"],

                    "rank": int(item["rank"]),
                    "gallery_relative_path": item["gallery_relative_path"],
                    "gallery_family_id": item["gallery_family_id"],
                    "gallery_subject_id": item["gallery_subject_id"],
                    "gallery_iris_id": item["gallery_iris_id"],
                    "gallery_eye": item["gallery_eye"],

                    "score": float(item["score"]),
                    "relation_to_probe": item["relation_to_probe"],
                    "is_true": bool(item["is_true"]),
                    "is_twin": bool(item["is_twin"]),
                    "is_twin_same_eye": bool(item["is_twin_same_eye"]),
                    "is_twin_cross_eye": bool(item["is_twin_cross_eye"]),
                    "common_bits": int(item["common_bits"]),
                    "best_shift": item["best_shift"],
                }
            )

        if (p_idx + 1) % 100 == 0:
            print(
                f"Probe processate {p_idx + 1}/{len(probe_rows)} "
                f"({comparison_counter}/{total_comparisons} confronti)"
            )

    results_df = pd.DataFrame(results)
    topk_df = pd.DataFrame(topk_rows)

    results_df.to_csv(results_csv, index=False)
    topk_df.to_csv(topk_csv_gz, index=False, compression="gzip")

    cmc_df = make_cmc(results_df, max_rank=cmc_max_rank)
    cmc_df.to_csv(cmc_csv, index=False)

    plot_cmc(cmc_df, plots_dir)
    plot_rank_histogram(results_df, plots_dir)
    plot_margin_histogram(results_df, plots_dir)

    n_probes = len(results_df)
    rank1_correct = int(results_df["rank1_correct"].sum())
    rank1_rate = rank1_correct / n_probes if n_probes else 0.0

    rank1_errors_df = results_df[results_df["rank1_correct"] == False].copy()
    rank1_errors = int(len(rank1_errors_df))

    probes_with_twin = results_df["min_twin_rank"].notna()
    n_probes_with_twin = int(probes_with_twin.sum())

    summary = {
        "input_features": project_display_path(input_features),
        "output_dir": project_display_path(output_dir),

        "input_feature_images": int(len(df)),
        "gallery_templates": int(len(gallery_rows)),
        "probe_templates": int(len(probe_rows)),
        "gallery_selection": gallery_selection,
        "seed": int(seed),

        "max_shift": int(max_shift),
        "shift_step": int(shift_step),
        "shifts": [int(s) for s in shifts],
        "min_common_bits": int(min_common_bits),

        "total_probe_gallery_comparisons": int(total_comparisons),
        "failed_probe_gallery_comparisons": int(failed_comparisons),

        "rank1_correct": int(rank1_correct),
        "rank1_errors": int(rank1_errors),
        "rank1_recognition_rate": float(rank1_rate),

        "CMS": {
            "CMS@1": float((results_df["true_rank"] <= 1).mean()),
            "CMS@2": float((results_df["true_rank"] <= 2).mean()),
            "CMS@5": float((results_df["true_rank"] <= 5).mean()),
            "CMS@10": float((results_df["true_rank"] <= 10).mean()),
            "CMS@20": float((results_df["true_rank"] <= 20).mean()),
        },

        "true_rank_stats": compute_stats(results_df["true_rank"]),
        "true_score_stats": compute_stats(results_df["true_score"]),
        "best_impostor_score_stats": compute_stats(results_df["best_impostor_score"]),
        "margin_best_impostor_minus_true_stats": compute_stats(results_df["margin_best_impostor_minus_true"]),

        "twin_analysis": {
            "probes_with_twin_in_gallery": n_probes_with_twin,
            "probes_with_twin_in_gallery_rate": pct(n_probes_with_twin, n_probes),

            "best_impostor_is_twin_count": int(results_df["best_impostor_is_twin"].sum()),
            "best_impostor_is_twin_rate": pct(int(results_df["best_impostor_is_twin"].sum()), n_probes),

            "best_impostor_is_twin_same_eye_count": int(results_df["best_impostor_is_twin_same_eye"].sum()),
            "best_impostor_is_twin_same_eye_rate": pct(int(results_df["best_impostor_is_twin_same_eye"].sum()), n_probes),

            "best_impostor_is_twin_cross_eye_count": int(results_df["best_impostor_is_twin_cross_eye"].sum()),
            "best_impostor_is_twin_cross_eye_rate": pct(int(results_df["best_impostor_is_twin_cross_eye"].sum()), n_probes),

            "rank1_error_is_twin_count": int(rank1_errors_df["rank1_is_twin"].sum()) if rank1_errors else 0,
            "rank1_error_is_twin_rate_among_errors": pct(
                int(rank1_errors_df["rank1_is_twin"].sum()) if rank1_errors else 0,
                rank1_errors,
            ),

            "twin_in_top2_count": int((results_df["min_twin_rank"] <= 2).sum()),
            "twin_in_top2_rate": pct(int((results_df["min_twin_rank"] <= 2).sum()), n_probes),

            "twin_in_top5_count": int((results_df["min_twin_rank"] <= 5).sum()),
            "twin_in_top5_rate": pct(int((results_df["min_twin_rank"] <= 5).sum()), n_probes),

            "twin_in_top10_count": int((results_df["min_twin_rank"] <= 10).sum()),
            "twin_in_top10_rate": pct(int((results_df["min_twin_rank"] <= 10).sum()), n_probes),

            "min_twin_rank_stats": compute_stats(results_df["min_twin_rank"]),
            "margin_best_twin_minus_true_stats": compute_stats(results_df["margin_best_twin_minus_true"]),
        },

        "outputs": {
            "results_csv": project_display_path(results_csv),
            "topk_csv_gz": project_display_path(topk_csv_gz),
            "cmc_csv": project_display_path(cmc_csv),
            "cmc_plot": project_display_path(plots_dir / "cmc_curve.png"),
            "true_rank_histogram": project_display_path(plots_dir / "true_rank_histogram.png"),
            "margin_histogram": project_display_path(plots_dir / "identification_margin_histogram.png"),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print()
    print("=== CLOSED-SET SUMMARY ===")
    print(f"Gallery templates:      {summary['gallery_templates']}")
    print(f"Probe templates:        {summary['probe_templates']}")
    print(f"Comparisons:            {summary['total_probe_gallery_comparisons']}")
    print(f"Failed comparisons:     {summary['failed_probe_gallery_comparisons']}")
    print()
    print(f"Rank-1 correct:         {summary['rank1_correct']}/{summary['probe_templates']}")
    print(f"Rank-1 recognition:     {summary['rank1_recognition_rate']:.4f}")
    print()
    print("CMS:")
    for k, v in summary["CMS"].items():
        print(f"  {k}: {v:.4f}")

    print()
    print("Twin analysis:")
    ta = summary["twin_analysis"]
    print(f"  Best impostor is twin:       {ta['best_impostor_is_twin_count']} ({ta['best_impostor_is_twin_rate']:.4f})")
    print(f"  Twin in top-2:               {ta['twin_in_top2_count']} ({ta['twin_in_top2_rate']:.4f})")
    print(f"  Twin in top-5:               {ta['twin_in_top5_count']} ({ta['twin_in_top5_rate']:.4f})")
    print(f"  Twin in top-10:              {ta['twin_in_top10_count']} ({ta['twin_in_top10_rate']:.4f})")
    print(f"  Rank-1 errors caused by twin:{ta['rank1_error_is_twin_count']} ({ta['rank1_error_is_twin_rate_among_errors']:.4f} of errors)")

    print()
    print("Output:")
    print(f"  Results: {project_display_path(results_csv)}")
    print(f"  Top-k:   {project_display_path(topk_csv_gz)}")
    print(f"  CMC:     {project_display_path(cmc_csv)}")
    print(f"  Summary: {project_display_path(summary_json)}")
    print(f"  Plots:   {project_display_path(plots_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-features",
        type=Path,
        default=DEFAULT_INPUT_FEATURES,
        help="Path a features_report.csv. Default: data/features/daugman_strict_v3/features_report.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella output closed-set. Default: data/identification_closed_set/daugman_strict_v3",
    )
    parser.add_argument("--max-shift", type=int, default=16)
    parser.add_argument("--shift-step", type=int, default=2)
    parser.add_argument("--min-common-bits", type=int, default=1000)
    parser.add_argument(
        "--gallery-selection",
        type=str,
        default="best_valid",
        choices=["best_valid", "first", "random"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cmc-max-rank", type=int, default=20)

    args = parser.parse_args()

    run_closed_set(
        input_features=resolve_from_project(args.input_features),
        output_dir=resolve_from_project(args.output_dir),
        max_shift=args.max_shift,
        shift_step=args.shift_step,
        min_common_bits=args.min_common_bits,
        gallery_selection=args.gallery_selection,
        seed=args.seed,
        top_k=args.top_k,
        cmc_max_rank=args.cmc_max_rank,
    )


if __name__ == "__main__":
    main()
