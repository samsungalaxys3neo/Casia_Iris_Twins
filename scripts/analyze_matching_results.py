from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_MATCHING = PROJECT_ROOT / "data" / "matching" / "daugman_strict_v3" / "matching_results.csv.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "daugman_strict_v3"

RELATION_ORDER = [
    "genuine",
    "same_subject_different_eye",
    "twin_same_eye",
    "twin_cross_eye",
    "unrelated",
]

IMPOSTOR_SETS = [
    None,
    "unrelated",
    "twin_same_eye",
    "twin_cross_eye",
    "same_subject_different_eye",
]


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


def read_matching_results(input_matching: Path) -> pd.DataFrame:
    dtype_map = {
        "pair_id": int,
        "sample_id_1": int,
        "sample_id_2": int,
        "image_1": str,
        "image_2": str,
        "code_path_1": str,
        "code_path_2": str,
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
        "relation": str,
    }

    df = pd.read_csv(input_matching, dtype=dtype_map)

    df["match_ok"] = (
        df["match_ok"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    df["hamming_distance"] = pd.to_numeric(df["hamming_distance"], errors="coerce")
    df["common_bits"] = pd.to_numeric(df["common_bits"], errors="coerce")
    df["best_shift"] = pd.to_numeric(df["best_shift"], errors="coerce")

    return df


def relation_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for relation in RELATION_ORDER:
        sub = df[df["relation"] == relation]["hamming_distance"].dropna()

        if len(sub) == 0:
            continue

        rows.append(
            {
                "relation": relation,
                "n": int(len(sub)),
                "min": float(sub.min()),
                "q1": float(sub.quantile(0.25)),
                "median": float(sub.median()),
                "mean": float(sub.mean()),
                "q3": float(sub.quantile(0.75)),
                "max": float(sub.max()),
                "std": float(sub.std()),
            }
        )

    return pd.DataFrame(rows)


def plot_histograms(df: pd.DataFrame, plots_dir: Path):
    plt.figure()
    for relation in ["genuine", "unrelated"]:
        values = df[df["relation"] == relation]["hamming_distance"].dropna()
        plt.hist(values, bins=60, alpha=0.55, density=True, label=relation)

    plt.xlabel("Hamming Distance")
    plt.ylabel("Density")
    plt.title("Hamming Distance: genuine vs unrelated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "hist_genuine_vs_unrelated.png", dpi=160)
    plt.close()

    plt.figure()
    for relation in ["genuine", "twin_same_eye", "unrelated"]:
        values = df[df["relation"] == relation]["hamming_distance"].dropna()
        plt.hist(values, bins=60, alpha=0.50, density=True, label=relation)

    plt.xlabel("Hamming Distance")
    plt.ylabel("Density")
    plt.title("Hamming Distance: genuine vs twin_same_eye vs unrelated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "hist_genuine_twin_unrelated.png", dpi=160)
    plt.close()


def plot_boxplot(df: pd.DataFrame, plots_dir: Path):
    data = []
    labels = []

    for relation in RELATION_ORDER:
        values = df[df["relation"] == relation]["hamming_distance"].dropna()
        if len(values) > 0:
            data.append(values)
            labels.append(relation)

    if not data:
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Hamming Distance")
    plt.title("Hamming Distance by relation")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "boxplot_by_relation.png", dpi=160)
    plt.close()


def qq_plot(
    df: pd.DataFrame,
    plots_dir: Path,
    relation_a: str = "twin_same_eye",
    relation_b: str = "unrelated",
):
    a = df[df["relation"] == relation_a]["hamming_distance"].dropna().to_numpy()
    b = df[df["relation"] == relation_b]["hamming_distance"].dropna().to_numpy()

    if len(a) == 0 or len(b) == 0:
        return

    qs = np.linspace(0.01, 0.99, 99)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)

    plt.figure()
    plt.scatter(qb, qa, s=12)
    min_v = min(qb.min(), qa.min())
    max_v = max(qb.max(), qa.max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel(f"{relation_b} quantiles")
    plt.ylabel(f"{relation_a} quantiles")
    plt.title(f"QQ plot: {relation_a} vs {relation_b}")
    plt.tight_layout()
    plt.savefig(plots_dir / f"qq_{relation_a}_vs_{relation_b}.png", dpi=160)
    plt.close()


def compute_verification_metrics(df: pd.DataFrame, impostor_relation=None):
    genuine = df[df["relation"] == "genuine"]["hamming_distance"].dropna().to_numpy()

    if impostor_relation is None:
        impostor = df[df["relation"] != "genuine"]["hamming_distance"].dropna().to_numpy()
        name = "all_impostor"
    else:
        impostor = df[df["relation"] == impostor_relation]["hamming_distance"].dropna().to_numpy()
        name = impostor_relation

    if len(genuine) == 0 or len(impostor) == 0:
        return None, None

    thresholds = np.linspace(0.0, 0.65, 651)

    rows = []

    for threshold in thresholds:
        # Accetto match se HD <= soglia.
        far = np.mean(impostor <= threshold)
        frr = np.mean(genuine > threshold)
        gar = 1.0 - frr
        grr = 1.0 - far

        rows.append(
            {
                "impostor_set": name,
                "threshold": float(threshold),
                "FAR": float(far),
                "FRR": float(frr),
                "GAR": float(gar),
                "GRR": float(grr),
                "abs_FAR_minus_FRR": float(abs(far - frr)),
            }
        )

    metrics = pd.DataFrame(rows)
    eer_row = metrics.iloc[metrics["abs_FAR_minus_FRR"].idxmin()]
    eer = float((eer_row["FAR"] + eer_row["FRR"]) / 2.0)

    roc = metrics.sort_values("FAR")
    auc = float(np.trapz(roc["GAR"], roc["FAR"]))

    summary = {
        "impostor_set": name,
        "n_genuine": int(len(genuine)),
        "n_impostor": int(len(impostor)),
        "EER": eer,
        "EER_threshold": float(eer_row["threshold"]),
        "FAR_at_EER": float(eer_row["FAR"]),
        "FRR_at_EER": float(eer_row["FRR"]),
        "AUC": auc,
    }

    return metrics, summary


def plot_roc(metrics_df: pd.DataFrame, output_path: Path):
    plt.figure()

    for impostor_set, group in metrics_df.groupby("impostor_set"):
        group = group.sort_values("FAR")
        plt.plot(group["FAR"], group["GAR"], label=impostor_set)

    plt.xlabel("FAR")
    plt.ylabel("GAR")
    plt.title("ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_det(metrics_df: pd.DataFrame, output_path: Path):
    plt.figure()

    for impostor_set, group in metrics_df.groupby("impostor_set"):
        group = group.sort_values("FAR")
        plt.plot(group["FAR"], group["FRR"], label=impostor_set)

    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("DET-style curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def cohen_d(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) < 2 or len(b) < 2:
        return None

    pooled = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )

    if pooled == 0:
        return None

    return float((np.mean(a) - np.mean(b)) / pooled)


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """
    Test descrittivi principali.

    Nota metodologica:
    le coppie non sono completamente indipendenti, perché una stessa immagine
    può comparire in più confronti. Quindi i p-value vanno interpretati con cautela.
    """
    results = {}

    try:
        from scipy.stats import ks_2samp, mannwhitneyu
        scipy_available = True
    except Exception:
        scipy_available = False

    comparisons = [
        ("genuine", "unrelated"),
        ("twin_same_eye", "unrelated"),
        ("twin_cross_eye", "unrelated"),
        ("same_subject_different_eye", "unrelated"),
        ("twin_same_eye", "twin_cross_eye"),
    ]

    for a_name, b_name in comparisons:
        a = df[df["relation"] == a_name]["hamming_distance"].dropna().to_numpy()
        b = df[df["relation"] == b_name]["hamming_distance"].dropna().to_numpy()

        item = {
            "a": a_name,
            "b": b_name,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_a": float(np.mean(a)) if len(a) else None,
            "mean_b": float(np.mean(b)) if len(b) else None,
            "median_a": float(np.median(a)) if len(a) else None,
            "median_b": float(np.median(b)) if len(b) else None,
            "mean_difference_a_minus_b": float(np.mean(a) - np.mean(b)) if len(a) and len(b) else None,
            "cohen_d_a_minus_b": cohen_d(a, b) if len(a) and len(b) else None,
            "independence_note": "Pairwise comparisons are not fully independent because images can appear in multiple pairs.",
        }

        if scipy_available and len(a) and len(b):
            ks = ks_2samp(a, b)
            mw = mannwhitneyu(a, b, alternative="two-sided")

            item["ks_statistic"] = float(ks.statistic)
            item["ks_pvalue"] = float(ks.pvalue)
            item["mannwhitney_u_statistic"] = float(mw.statistic)
            item["mannwhitney_pvalue"] = float(mw.pvalue)
        else:
            item["note"] = "scipy not available, only descriptive/effect-size metrics computed"

        results[f"{a_name}_vs_{b_name}"] = item

    return results


def run_analysis(input_matching: Path, output_dir: Path):
    if not input_matching.exists():
        raise FileNotFoundError(f"Matching results non trovato: {input_matching}")

    plots_dir = output_dir / "plots"

    summary_csv = output_dir / "matching_relation_stats.csv"
    verification_csv = output_dir / "verification_metrics.csv"
    stat_tests_json = output_dir / "statistical_tests.json"
    analysis_summary_json = output_dir / "analysis_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=== ANALYZE MATCHING RESULTS ===")
    print(f"Input:      {project_display_path(input_matching)}")
    print(f"Output dir: {project_display_path(output_dir)}")

    df = read_matching_results(input_matching)
    df = df[df["match_ok"] == True].copy()

    print(f"Match OK caricati: {len(df)}")

    stats_df = relation_stats(df)
    stats_df.to_csv(summary_csv, index=False)

    print("\n=== Relation stats ===")
    print(stats_df.to_string(index=False))

    plot_histograms(df, plots_dir)
    plot_boxplot(df, plots_dir)
    qq_plot(df, plots_dir, "twin_same_eye", "unrelated")
    qq_plot(df, plots_dir, "twin_cross_eye", "unrelated")

    all_verification = []
    verification_summaries = {}

    for impostor_relation in IMPOSTOR_SETS:
        metrics, summary = compute_verification_metrics(df, impostor_relation=impostor_relation)

        if metrics is not None:
            all_verification.append(metrics)
            verification_summaries[summary["impostor_set"]] = summary

    if all_verification:
        verification_df = pd.concat(all_verification, ignore_index=True)
        verification_df.to_csv(verification_csv, index=False)

        plot_roc(verification_df, plots_dir / "roc_curves.png")
        plot_det(verification_df, plots_dir / "det_curves.png")
    else:
        verification_df = pd.DataFrame()
        verification_df.to_csv(verification_csv, index=False)

    stat_tests = run_statistical_tests(df)

    with stat_tests_json.open("w", encoding="utf-8") as f:
        json.dump(stat_tests, f, indent=2)

    analysis_summary = {
        "input_matching_file": project_display_path(input_matching),
        "output_dir": project_display_path(output_dir),
        "n_match_ok": int(len(df)),
        "relation_stats": stats_df.to_dict(orient="records"),
        "verification_summary": verification_summaries,
        "statistical_tests": stat_tests,
        "methodological_note": "Statistical tests are reported as descriptive support. Pairwise samples are not fully independent because the same image can appear in multiple pairs.",
        "plots": {
            "hist_genuine_vs_unrelated": project_display_path(plots_dir / "hist_genuine_vs_unrelated.png"),
            "hist_genuine_twin_unrelated": project_display_path(plots_dir / "hist_genuine_twin_unrelated.png"),
            "boxplot_by_relation": project_display_path(plots_dir / "boxplot_by_relation.png"),
            "qq_twin_same_eye_vs_unrelated": project_display_path(plots_dir / "qq_twin_same_eye_vs_unrelated.png"),
            "qq_twin_cross_eye_vs_unrelated": project_display_path(plots_dir / "qq_twin_cross_eye_vs_unrelated.png"),
            "roc_curves": project_display_path(plots_dir / "roc_curves.png"),
            "det_curves": project_display_path(plots_dir / "det_curves.png"),
        },
    }

    with analysis_summary_json.open("w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2)

    print("\n=== Verification summary ===")
    for name, item in verification_summaries.items():
        print(
            f"{name:30s} "
            f"EER={item['EER']:.4f} "
            f"thr={item['EER_threshold']:.4f} "
            f"AUC={item['AUC']:.4f}"
        )

    print("\n=== Statistical tests summary ===")
    for name, item in stat_tests.items():
        print(name)
        print(f"  mean diff: {item['mean_difference_a_minus_b']}")
        print(f"  cohen d:   {item['cohen_d_a_minus_b']}")
        if "ks_pvalue" in item:
            print(f"  KS p:      {item['ks_pvalue']}")
            print(f"  MW p:      {item['mannwhitney_pvalue']}")

    print("\nOutput salvati:")
    print(f"  Relation stats:       {project_display_path(summary_csv)}")
    print(f"  Verification metrics: {project_display_path(verification_csv)}")
    print(f"  Statistical tests:    {project_display_path(stat_tests_json)}")
    print(f"  Analysis summary:     {project_display_path(analysis_summary_json)}")
    print(f"  Plots:                {project_display_path(plots_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-matching",
        type=Path,
        default=DEFAULT_INPUT_MATCHING,
        help="Path a matching_results.csv.gz. Default: data/matching/daugman_strict_v3/matching_results.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella output analysis. Default: data/analysis/daugman_strict_v3",
    )

    args = parser.parse_args()

    run_analysis(
        input_matching=resolve_from_project(args.input_matching),
        output_dir=resolve_from_project(args.output_dir),
    )


if __name__ == "__main__":
    main()
