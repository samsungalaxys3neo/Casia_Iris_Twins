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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "statistical_resampling" / "daugman_strict_v3"


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


def as_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_number(value):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
        return None
    except Exception:
        return None


def safe_filename(name):
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return None

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    pooled = math.sqrt(
        ((len(a) - 1) * var_a + (len(b) - 1) * var_b)
        / (len(a) + len(b) - 2)
    )

    if pooled == 0:
        return None

    return safe_number((np.mean(a) - np.mean(b)) / pooled)


def bootstrap_mean_difference(a, b, n_bootstrap, rng):
    """
    Bootstrap indipendente sui due gruppi.
    Ritorna distribuzione bootstrap di mean(a*) - mean(b*).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n_a = len(a)
    n_b = len(b)

    diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        a_sample = a[rng.integers(0, n_a, size=n_a)]
        b_sample = b[rng.integers(0, n_b, size=n_b)]
        diffs[i] = np.mean(a_sample) - np.mean(b_sample)

    return diffs


def permutation_test_mean_difference(a, b, n_permutation, rng):
    """
    Permutation test two-sided sulla differenza media.
    H0: i due gruppi sono scambiabili.
    Statistica: mean(a) - mean(b).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n_a = len(a)
    n_b = len(b)

    observed = float(np.mean(a) - np.mean(b))

    pooled = np.concatenate([a, b])
    total_sum = float(np.sum(pooled))
    n_total = len(pooled)

    extreme = 0

    for _ in range(n_permutation):
        idx_a = rng.choice(n_total, size=n_a, replace=False)

        sum_a = float(np.sum(pooled[idx_a]))
        mean_a = sum_a / n_a
        mean_b = (total_sum - sum_a) / n_b

        perm_diff = mean_a - mean_b

        if abs(perm_diff) >= abs(observed):
            extreme += 1

    # Correzione +1 per evitare p-value esattamente zero.
    p_value = (extreme + 1) / (n_permutation + 1)

    return observed, float(p_value)


def summarize_scores(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    return {
        "n": int(len(values)),
        "min": safe_number(np.min(values)),
        "q1": safe_number(np.quantile(values, 0.25)),
        "median": safe_number(np.median(values)),
        "mean": safe_number(np.mean(values)),
        "q3": safe_number(np.quantile(values, 0.75)),
        "max": safe_number(np.max(values)),
        "std": safe_number(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def plot_bootstrap_distribution(
    plots_dir: Path,
    comparison_name: str,
    boot_diffs,
    observed_diff,
    ci_low,
    ci_high,
):
    plt.figure()
    plt.hist(boot_diffs, bins=60, density=True, alpha=0.75)
    plt.axvline(observed_diff, linestyle="-", label="observed mean diff")
    plt.axvline(ci_low, linestyle="--", label="95% CI low")
    plt.axvline(ci_high, linestyle="--", label="95% CI high")
    plt.axvline(0, linestyle=":", label="zero difference")

    plt.xlabel("Mean difference")
    plt.ylabel("Density")
    plt.title(comparison_name)
    plt.legend()
    plt.tight_layout()

    output_path = plots_dir / f"bootstrap_{safe_filename(comparison_name)}.png"
    plt.savefig(output_path, dpi=160)
    plt.close()

    return output_path


def get_group_scores(df, group_name):
    if group_name == "genuine":
        sub = df[df["relation"] == "genuine"]

    elif group_name == "same_subject_different_eye":
        sub = df[df["relation"] == "same_subject_different_eye"]

    elif group_name == "twin_same_eye":
        sub = df[df["relation"] == "twin_same_eye"]

    elif group_name == "twin_cross_eye":
        sub = df[df["relation"] == "twin_cross_eye"]

    elif group_name == "twin_all":
        sub = df[df["relation"].isin(["twin_same_eye", "twin_cross_eye"])]

    elif group_name == "unrelated":
        sub = df[df["relation"] == "unrelated"]

    elif group_name == "unrelated_same_eye":
        sub = df[
            (df["relation"] == "unrelated")
            & (df["eye_1"].astype(str) == df["eye_2"].astype(str))
        ]

    elif group_name == "unrelated_cross_eye":
        sub = df[
            (df["relation"] == "unrelated")
            & (df["eye_1"].astype(str) != df["eye_2"].astype(str))
        ]

    elif group_name == "all_impostor":
        sub = df[df["relation"] != "genuine"]

    else:
        raise ValueError(f"Unknown group: {group_name}")

    return pd.to_numeric(sub["hamming_distance"], errors="coerce").dropna().to_numpy()


def interpretation_from_ci(ci_low, ci_high):
    if ci_high < 0:
        direction = "group_a_lower"
        text = "Group A has lower Hamming Distance than Group B."
    elif ci_low > 0:
        direction = "group_a_higher"
        text = "Group A has higher Hamming Distance than Group B."
    else:
        direction = "ci_crosses_zero"
        text = "The confidence interval crosses zero."

    return direction, text


def run_comparison(df, comparison, n_bootstrap, n_permutation, seed, plots_dir):
    rng = np.random.default_rng(seed)

    name = comparison["name"]
    group_a = comparison["group_a"]
    group_b = comparison["group_b"]

    a = get_group_scores(df, group_a)
    b = get_group_scores(df, group_b)

    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"Too few samples for comparison {name}")

    observed_diff, permutation_p = permutation_test_mean_difference(
        a,
        b,
        n_permutation=n_permutation,
        rng=rng,
    )

    boot_diffs = bootstrap_mean_difference(
        a,
        b,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    ci_low = float(np.quantile(boot_diffs, 0.025))
    ci_high = float(np.quantile(boot_diffs, 0.975))
    boot_mean = float(np.mean(boot_diffs))
    boot_std = float(np.std(boot_diffs, ddof=1))

    effect = cohen_d(a, b)
    direction, interpretation = interpretation_from_ci(ci_low, ci_high)

    plot_path = plot_bootstrap_distribution(
        plots_dir=plots_dir,
        comparison_name=name,
        boot_diffs=boot_diffs,
        observed_diff=observed_diff,
        ci_low=ci_low,
        ci_high=ci_high,
    )

    result = {
        "comparison": name,
        "group_a": group_a,
        "group_b": group_b,

        "n_a": int(len(a)),
        "n_b": int(len(b)),

        "mean_a": safe_number(np.mean(a)),
        "mean_b": safe_number(np.mean(b)),
        "median_a": safe_number(np.median(a)),
        "median_b": safe_number(np.median(b)),
        "std_a": safe_number(np.std(a, ddof=1)),
        "std_b": safe_number(np.std(b, ddof=1)),

        "observed_mean_difference_a_minus_b": safe_number(observed_diff),

        "bootstrap_n": int(n_bootstrap),
        "bootstrap_mean_difference_mean": safe_number(boot_mean),
        "bootstrap_mean_difference_std": safe_number(boot_std),
        "bootstrap_ci_95_low": safe_number(ci_low),
        "bootstrap_ci_95_high": safe_number(ci_high),

        "permutation_n": int(n_permutation),
        "permutation_pvalue_two_sided": safe_number(permutation_p),

        "cohen_d_a_minus_b": effect,

        "ci_direction": direction,
        "interpretation": interpretation,

        "bootstrap_plot": project_display_path(plot_path),
    }

    return result


def read_matching(input_matching: Path) -> pd.DataFrame:
    df = pd.read_csv(
        input_matching,
        dtype={
            "eye_1": str,
            "eye_2": str,
            "relation": str,
        },
    )

    df = df[as_bool(df["match_ok"])].copy()
    df["hamming_distance"] = pd.to_numeric(df["hamming_distance"], errors="coerce")
    df = df.dropna(subset=["hamming_distance"])

    return df


def run_resampling(
    input_matching: Path,
    output_dir: Path,
    n_bootstrap: int,
    n_permutation: int,
    seed: int,
):
    if not input_matching.exists():
        raise FileNotFoundError(f"Matching results non trovato: {input_matching}")

    plots_dir = output_dir / "plots"
    results_csv = output_dir / "bootstrap_permutation_results.csv"
    summary_json = output_dir / "bootstrap_permutation_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=== BOOTSTRAP + PERMUTATION TESTS ===")
    print(f"Input:          {project_display_path(input_matching)}")
    print(f"Output dir:     {project_display_path(output_dir)}")
    print(f"Bootstrap:      {n_bootstrap}")
    print(f"Permutation:    {n_permutation}")
    print(f"Seed:           {seed}")
    print()

    df = read_matching(input_matching)

    comparisons = [
        {
            "name": "genuine_vs_unrelated",
            "group_a": "genuine",
            "group_b": "unrelated",
        },
        {
            "name": "twin_all_vs_unrelated",
            "group_a": "twin_all",
            "group_b": "unrelated",
        },
        {
            "name": "twin_same_eye_vs_unrelated_same_eye",
            "group_a": "twin_same_eye",
            "group_b": "unrelated_same_eye",
        },
        {
            "name": "twin_cross_eye_vs_unrelated_cross_eye",
            "group_a": "twin_cross_eye",
            "group_b": "unrelated_cross_eye",
        },
        {
            "name": "same_subject_different_eye_vs_unrelated",
            "group_a": "same_subject_different_eye",
            "group_b": "unrelated",
        },
        {
            "name": "twin_same_eye_vs_twin_cross_eye",
            "group_a": "twin_same_eye",
            "group_b": "twin_cross_eye",
        },
    ]

    group_names = [
        "genuine",
        "same_subject_different_eye",
        "twin_same_eye",
        "twin_cross_eye",
        "twin_all",
        "unrelated",
        "unrelated_same_eye",
        "unrelated_cross_eye",
        "all_impostor",
    ]

    group_summary = {}

    for group in group_names:
        values = get_group_scores(df, group)
        group_summary[group] = summarize_scores(values)

    results = []

    for i, comparison in enumerate(comparisons):
        print(f"Running {comparison['name']}...")

        result = run_comparison(
            df=df,
            comparison=comparison,
            n_bootstrap=n_bootstrap,
            n_permutation=n_permutation,
            seed=seed + i,
            plots_dir=plots_dir,
        )

        results.append(result)

        print(
            f"  diff={result['observed_mean_difference_a_minus_b']:.6f} "
            f"CI=[{result['bootstrap_ci_95_low']:.6f}, {result['bootstrap_ci_95_high']:.6f}] "
            f"p={result['permutation_pvalue_two_sided']:.6g} "
            f"d={result['cohen_d_a_minus_b']:.4f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)

    summary = {
        "input_file": project_display_path(input_matching),
        "output_dir": project_display_path(output_dir),
        "n_matching_rows": int(len(df)),
        "n_bootstrap": int(n_bootstrap),
        "n_permutation": int(n_permutation),
        "seed": int(seed),
        "group_summary": group_summary,
        "comparisons": results,
        "outputs": {
            "results_csv": project_display_path(results_csv),
            "summary_json": project_display_path(summary_json),
            "plots_dir": project_display_path(plots_dir),
        },
        "note": (
            "Bootstrap and permutation tests are computed at the pair level. "
            "Because pairwise biometric comparisons are not fully independent, "
            "p-values should be interpreted cautiously and together with effect sizes "
            "and confidence intervals."
        ),
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print()
    print("=== DONE ===")
    print(f"Results CSV: {project_display_path(results_csv)}")
    print(f"Summary:     {project_display_path(summary_json)}")
    print(f"Plots:       {project_display_path(plots_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-matching",
        type=Path,
        default=DEFAULT_INPUT_MATCHING,
        help="Path a matching_results.csv.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella output statistical resampling.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutation", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_resampling(
        input_matching=resolve_from_project(args.input_matching),
        output_dir=resolve_from_project(args.output_dir),
        n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
