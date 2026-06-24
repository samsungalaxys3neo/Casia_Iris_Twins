from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_MATCHING = PROJECT_ROOT / "data/matching/daugman_strict_v3/matching_results.csv.gz"

OUTPUT_DIR = PROJECT_ROOT / "data/analysis/daugman_strict_v3"
PLOTS_DIR = OUTPUT_DIR / "plots"

SUMMARY_CSV = OUTPUT_DIR / "matching_relation_stats.csv"
VERIFICATION_CSV = OUTPUT_DIR / "verification_metrics.csv"
STAT_TESTS_JSON = OUTPUT_DIR / "statistical_tests.json"
ANALYSIS_SUMMARY_JSON = OUTPUT_DIR / "analysis_summary.json"


RELATION_ORDER = [
    "genuine",
    "same_subject_different_eye",
    "twin_same_eye",
    "twin_cross_eye",
    "unrelated",
]


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def relation_stats(df):
    rows = []

    for relation in RELATION_ORDER:
        sub = df[df["relation"] == relation]["hamming_distance"].dropna()

        if len(sub) == 0:
            continue

        rows.append({
            "relation": relation,
            "n": int(len(sub)),
            "min": float(sub.min()),
            "q1": float(sub.quantile(0.25)),
            "median": float(sub.median()),
            "mean": float(sub.mean()),
            "q3": float(sub.quantile(0.75)),
            "max": float(sub.max()),
            "std": float(sub.std()),
        })

    return pd.DataFrame(rows)


def plot_histograms(df):
    # Genuine vs unrelated
    plt.figure()
    for relation in ["genuine", "unrelated"]:
        values = df[df["relation"] == relation]["hamming_distance"].dropna()
        plt.hist(values, bins=60, alpha=0.55, density=True, label=relation)
    plt.xlabel("Hamming Distance")
    plt.ylabel("Density")
    plt.title("Hamming Distance: genuine vs unrelated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hist_genuine_vs_unrelated.png")
    plt.close()

    # Genuine vs twin same-eye vs unrelated
    plt.figure()
    for relation in ["genuine", "twin_same_eye", "unrelated"]:
        values = df[df["relation"] == relation]["hamming_distance"].dropna()
        plt.hist(values, bins=60, alpha=0.50, density=True, label=relation)
    plt.xlabel("Hamming Distance")
    plt.ylabel("Density")
    plt.title("Hamming Distance: genuine vs twin_same_eye vs unrelated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hist_genuine_twin_unrelated.png")
    plt.close()


def plot_boxplot(df):
    data = []
    labels = []

    for relation in RELATION_ORDER:
        values = df[df["relation"] == relation]["hamming_distance"].dropna()
        if len(values) > 0:
            data.append(values)
            labels.append(relation)

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Hamming Distance")
    plt.title("Hamming Distance by relation")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "boxplot_by_relation.png")
    plt.close()


def qq_plot(df, relation_a="twin_same_eye", relation_b="unrelated"):
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
    plt.savefig(PLOTS_DIR / f"qq_{relation_a}_vs_{relation_b}.png")
    plt.close()


def compute_verification_metrics(df, impostor_relation=None):
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

    for t in thresholds:
        # Accetto match se HD <= soglia.
        far = np.mean(impostor <= t)
        frr = np.mean(genuine > t)
        gar = 1.0 - frr
        grr = 1.0 - far

        rows.append({
            "impostor_set": name,
            "threshold": float(t),
            "FAR": float(far),
            "FRR": float(frr),
            "GAR": float(gar),
            "GRR": float(grr),
            "abs_FAR_minus_FRR": float(abs(far - frr)),
        })

    metrics = pd.DataFrame(rows)
    eer_row = metrics.iloc[metrics["abs_FAR_minus_FRR"].idxmin()]
    eer = float((eer_row["FAR"] + eer_row["FRR"]) / 2.0)

    # AUC ROC: x = FAR, y = GAR
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


def plot_roc(metrics_df, output_path):
    plt.figure()

    for impostor_set, group in metrics_df.groupby("impostor_set"):
        group = group.sort_values("FAR")
        plt.plot(group["FAR"], group["GAR"], label=impostor_set)

    plt.xlabel("FAR")
    plt.ylabel("GAR")
    plt.title("ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_det(metrics_df, output_path):
    plt.figure()

    for impostor_set, group in metrics_df.groupby("impostor_set"):
        group = group.sort_values("FAR")
        plt.plot(group["FAR"], group["FRR"], label=impostor_set)

    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("DET-style curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def cohen_d(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )

    if pooled == 0:
        return np.nan

    return float((np.mean(a) - np.mean(b)) / pooled)


def run_statistical_tests(df):
    """
    Test principali:
    - twin_same_eye vs unrelated
    - twin_cross_eye vs unrelated
    - same_subject_different_eye vs unrelated
    - genuine vs unrelated

    Usa scipy se disponibile.
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


def main():
    ensure_dirs()

    print("=== ANALYZE MATCHING RESULTS ===")
    print(f"Input: {INPUT_MATCHING}")

    df = pd.read_csv(INPUT_MATCHING)
    df = df[df["match_ok"] == True].copy()
    df["hamming_distance"] = pd.to_numeric(df["hamming_distance"], errors="coerce")

    print(f"Match OK caricati: {len(df)}")

    stats_df = relation_stats(df)
    stats_df.to_csv(SUMMARY_CSV, index=False)

    print("\n=== Relation stats ===")
    print(stats_df.to_string(index=False))

    plot_histograms(df)
    plot_boxplot(df)
    qq_plot(df, "twin_same_eye", "unrelated")
    qq_plot(df, "twin_cross_eye", "unrelated")

    all_verification = []
    verification_summaries = {}

    for impostor_relation in [None, "unrelated", "twin_same_eye", "twin_cross_eye", "same_subject_different_eye"]:
        metrics, summary = compute_verification_metrics(df, impostor_relation=impostor_relation)

        if metrics is not None:
            all_verification.append(metrics)
            verification_summaries[summary["impostor_set"]] = summary

    verification_df = pd.concat(all_verification, ignore_index=True)
    verification_df.to_csv(VERIFICATION_CSV, index=False)

    plot_roc(verification_df, PLOTS_DIR / "roc_curves.png")
    plot_det(verification_df, PLOTS_DIR / "det_curves.png")

    stat_tests = run_statistical_tests(df)

    with STAT_TESTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(stat_tests, f, indent=2)

    analysis_summary = {
        "input_matching_file": str(INPUT_MATCHING),
        "n_match_ok": int(len(df)),
        "relation_stats": stats_df.to_dict(orient="records"),
        "verification_summary": verification_summaries,
        "statistical_tests": stat_tests,
        "plots": {
            "hist_genuine_vs_unrelated": str(PLOTS_DIR / "hist_genuine_vs_unrelated.png"),
            "hist_genuine_twin_unrelated": str(PLOTS_DIR / "hist_genuine_twin_unrelated.png"),
            "boxplot_by_relation": str(PLOTS_DIR / "boxplot_by_relation.png"),
            "qq_twin_same_eye_vs_unrelated": str(PLOTS_DIR / "qq_twin_same_eye_vs_unrelated.png"),
            "qq_twin_cross_eye_vs_unrelated": str(PLOTS_DIR / "qq_twin_cross_eye_vs_unrelated.png"),
            "roc_curves": str(PLOTS_DIR / "roc_curves.png"),
            "det_curves": str(PLOTS_DIR / "det_curves.png"),
        }
    }

    with ANALYSIS_SUMMARY_JSON.open("w", encoding="utf-8") as f:
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
    print(f"  Relation stats:       {SUMMARY_CSV}")
    print(f"  Verification metrics: {VERIFICATION_CSV}")
    print(f"  Statistical tests:    {STAT_TESTS_JSON}")
    print(f"  Analysis summary:     {ANALYSIS_SUMMARY_JSON}")
    print(f"  Plots:                {PLOTS_DIR}")


if __name__ == "__main__":
    main()