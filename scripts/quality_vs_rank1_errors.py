from pathlib import Path
import argparse
import json
import math

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLOSED_SET_PATH = (
    PROJECT_ROOT
    / "data"
    / "identification_closed_set"
    / "daugman_strict_v3"
    / "closed_set_results.csv"
)

DEFAULT_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "daugman_strict_v3"
    / "features_report.csv"
)

DEFAULT_QUALITY_PATH = PROJECT_ROOT / "data" / "quality" / "quality_report.csv"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "failure_analysis" / "daugman_strict_v3"


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


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def summarize_metric(df: pd.DataFrame, metric: str):
    correct = safe_numeric(df[df["rank1_correct"] == True][metric])
    errors = safe_numeric(df[df["rank1_correct"] == False][metric])

    if len(correct) == 0 or len(errors) == 0:
        return None

    return {
        "metric": metric,
        "correct_n": int(len(correct)),
        "error_n": int(len(errors)),
        "correct_mean": float(correct.mean()),
        "error_mean": float(errors.mean()),
        "difference_error_minus_correct": float(errors.mean() - correct.mean()),
        "correct_median": float(correct.median()),
        "error_median": float(errors.median()),
    }


def compute_effect_size(correct: pd.Series, errors: pd.Series):
    """
    Cohen-like standardized difference:
    effect_size = (error_mean - correct_mean) / pooled_std

    Negative value: metric is lower in Rank-1 errors.
    Positive value: metric is higher in Rank-1 errors.
    """
    correct = safe_numeric(correct)
    errors = safe_numeric(errors)

    if len(correct) < 2 or len(errors) < 2:
        return None

    correct_std = correct.std()
    error_std = errors.std()

    if pd.isna(correct_std) or pd.isna(error_std):
        return None

    pooled_var = (
        ((len(correct) - 1) * correct_std**2)
        + ((len(errors) - 1) * error_std**2)
    ) / (len(correct) + len(errors) - 2)

    pooled_std = math.sqrt(pooled_var)

    if pooled_std == 0:
        return None

    return float((errors.mean() - correct.mean()) / pooled_std)


def save_metric_boxplot(df: pd.DataFrame, metric: str, output_path: Path):
    correct = safe_numeric(df[df["rank1_correct"] == True][metric])
    errors = safe_numeric(df[df["rank1_correct"] == False][metric])

    if len(correct) == 0 or len(errors) == 0:
        return

    plt.figure(figsize=(5, 4))
    plt.boxplot(
        [correct, errors],
        labels=["Rank-1 correct", "Rank-1 errors"],
        showfliers=False,
    )
    plt.ylabel(metric)
    plt.title(f"{metric}: correct vs errors")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_valid_bit_ratio_histogram(df: pd.DataFrame, output_path: Path):
    correct = safe_numeric(df[df["rank1_correct"] == True]["valid_bit_ratio"])
    errors = safe_numeric(df[df["rank1_correct"] == False]["valid_bit_ratio"])

    if len(correct) == 0 or len(errors) == 0:
        return

    plt.figure(figsize=(6, 4))
    plt.hist(correct, bins=35, alpha=0.6, density=True, label="Rank-1 correct")
    plt.hist(errors, bins=35, alpha=0.6, density=True, label="Rank-1 errors")
    plt.xlabel("valid_bit_ratio")
    plt.ylabel("Density")
    plt.title("Valid-bit ratio distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_scatter(df: pd.DataFrame, x_metric: str, y_metric: str, output_path: Path):
    if x_metric not in df.columns or y_metric not in df.columns:
        return

    correct = df[df["rank1_correct"] == True].copy()
    errors = df[df["rank1_correct"] == False].copy()

    plt.figure(figsize=(6, 4))

    plt.scatter(
        pd.to_numeric(correct[x_metric], errors="coerce"),
        pd.to_numeric(correct[y_metric], errors="coerce"),
        alpha=0.5,
        s=16,
        label="Rank-1 correct",
    )

    plt.scatter(
        pd.to_numeric(errors[x_metric], errors="coerce"),
        pd.to_numeric(errors[y_metric], errors="coerce"),
        alpha=0.7,
        s=16,
        label="Rank-1 errors",
    )

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f"{y_metric} vs {x_metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_quality_effect_size_plot(df: pd.DataFrame, output_path: Path):
    """
    This plot compares all quality indicators on the same standardized scale.
    It is useful because brightness, focus_score, valid_bit_ratio, etc.
    have very different numerical ranges.
    """
    metrics = [
        "valid_bit_ratio",
        "bit_one_ratio",
        "brightness",
        "contrast",
        "focus_score",
        "dark_ratio",
        "bright_ratio",
    ]

    rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        correct = df[df["rank1_correct"] == True][metric]
        errors = df[df["rank1_correct"] == False][metric]

        effect_size = compute_effect_size(correct, errors)

        if effect_size is None:
            continue

        rows.append(
            {
                "metric": metric,
                "effect_size": effect_size,
            }
        )

    if not rows:
        return

    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.sort_values("effect_size")

    plt.figure(figsize=(7, 4.5))
    plt.barh(plot_df["metric"], plot_df["effect_size"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Standardized difference: Rank-1 errors - Rank-1 correct")
    plt.title("Quality indicators: errors vs correct cases")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_quality_plots(df: pd.DataFrame, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "valid_bit_ratio",
        "bit_one_ratio",
        "brightness",
        "contrast",
        "focus_score",
        "dark_ratio",
        "bright_ratio",
    ]

    for metric in metrics:
        if metric in df.columns:
            save_metric_boxplot(
                df,
                metric,
                plots_dir / f"boxplot_{metric}.png",
            )

    if "valid_bit_ratio" in df.columns:
        save_valid_bit_ratio_histogram(
            df,
            plots_dir / "hist_valid_bit_ratio.png",
        )

    save_scatter(
        df,
        "valid_bit_ratio",
        "margin_best_impostor_minus_true",
        plots_dir / "scatter_valid_bit_ratio_vs_margin.png",
    )

    save_scatter(
        df,
        "valid_bit_ratio",
        "true_rank",
        plots_dir / "scatter_valid_bit_ratio_vs_true_rank.png",
    )

    save_quality_effect_size_plot(
        df,
        plots_dir / "bar_quality_effect_sizes.png",
    )


def run_quality_vs_rank1_errors(
    closed_set_path: Path,
    features_path: Path,
    quality_path: Path,
    output_dir: Path,
):
    closed_set_path = resolve_from_project(closed_set_path)
    features_path = resolve_from_project(features_path)
    quality_path = resolve_from_project(quality_path)
    output_dir = resolve_from_project(output_dir)

    if not closed_set_path.exists():
        raise FileNotFoundError(f"Closed-set results not found: {closed_set_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Features report not found: {features_path}")

    if not quality_path.exists():
        raise FileNotFoundError(f"Quality report not found: {quality_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "quality_vs_rank1_errors.csv"
    output_json = output_dir / "quality_vs_rank1_errors_summary.json"
    output_probe_csv = output_dir / "quality_vs_rank1_probe_level.csv"
    plots_dir = output_dir / "quality_vs_rank1_plots"

    closed = pd.read_csv(
        closed_set_path,
        dtype={
            "probe_relative_path": str,
            "probe_iris_id": str,
        },
    )

    features = pd.read_csv(
        features_path,
        dtype={
            "relative_path": str,
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "subject_id": str,
            "iris_id": str,
        },
    )

    quality = pd.read_csv(
        quality_path,
        dtype={
            "relative_path": str,
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "subject_id": str,
            "iris_id": str,
        },
    )

    closed["rank1_correct"] = as_bool(closed["rank1_correct"])

    feature_cols = [
        "relative_path",
        "valid_bit_ratio",
        "bit_one_ratio",
    ]
    feature_cols = [c for c in feature_cols if c in features.columns]

    quality_cols = [
        "relative_path",
        "brightness",
        "contrast",
        "focus_score",
        "dark_ratio",
        "bright_ratio",
    ]
    quality_cols = [c for c in quality_cols if c in quality.columns]

    features_small = features[feature_cols].copy()
    features_small = features_small.rename(columns={"relative_path": "probe_relative_path"})

    quality_small = quality[quality_cols].copy()
    quality_small = quality_small.rename(columns={"relative_path": "probe_relative_path"})

    df = closed.merge(
        features_small,
        on="probe_relative_path",
        how="left",
    )

    df = df.merge(
        quality_small,
        on="probe_relative_path",
        how="left",
    )

    metrics = [
        "valid_bit_ratio",
        "bit_one_ratio",
        "brightness",
        "contrast",
        "focus_score",
        "dark_ratio",
        "bright_ratio",
        "margin_best_impostor_minus_true",
        "true_rank",
    ]

    rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        row = summarize_metric(df, metric)
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)

    out.to_csv(output_csv, index=False)
    df.to_csv(output_probe_csv, index=False)
    save_quality_plots(df, plots_dir)

    effect_sizes = {}

    for metric in [
        "valid_bit_ratio",
        "bit_one_ratio",
        "brightness",
        "contrast",
        "focus_score",
        "dark_ratio",
        "bright_ratio",
    ]:
        if metric not in df.columns:
            continue

        effect = compute_effect_size(
            df[df["rank1_correct"] == True][metric],
            df[df["rank1_correct"] == False][metric],
        )

        if effect is not None:
            effect_sizes[metric] = effect

    summary = {
        "inputs": {
            "closed_set_results": project_display_path(closed_set_path),
            "features_report": project_display_path(features_path),
            "quality_report": project_display_path(quality_path),
        },
        "outputs": {
            "quality_vs_rank1_errors_csv": project_display_path(output_csv),
            "quality_vs_rank1_probe_level_csv": project_display_path(output_probe_csv),
            "plots_dir": project_display_path(plots_dir),
        },
        "counts": {
            "closed_set_probes": int(len(closed)),
            "rank1_correct": int(closed["rank1_correct"].sum()),
            "rank1_errors": int((closed["rank1_correct"] == False).sum()),
        },
        "effect_sizes_error_minus_correct": effect_sizes,
        "main_interpretation": (
            "This analysis compares correctly identified probes with Rank-1 errors. "
            "The most relevant indicator is valid_bit_ratio: lower values suggest "
            "less usable iris texture after masking. Brightness, dark_ratio and "
            "bright_ratio are included to check whether errors are simply explained "
            "by exposure or illumination problems."
        ),
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== QUALITY VS RANK-1 ERRORS ===")
    print(f"Closed-set results: {project_display_path(closed_set_path)}")
    print(f"Features report:    {project_display_path(features_path)}")
    print(f"Quality report:     {project_display_path(quality_path)}")
    print(f"Output CSV:         {project_display_path(output_csv)}")
    print(f"Probe-level CSV:    {project_display_path(output_probe_csv)}")
    print(f"Output summary:     {project_display_path(output_json)}")
    print(f"Plots dir:          {project_display_path(plots_dir)}")
    print()
    print(out.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closed-set",
        type=Path,
        default=DEFAULT_CLOSED_SET_PATH,
        help="Path to closed_set_results.csv.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to features_report.csv.",
    )
    parser.add_argument(
        "--quality",
        type=Path,
        default=DEFAULT_QUALITY_PATH,
        help="Path to quality_report.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )

    args = parser.parse_args()

    run_quality_vs_rank1_errors(
        closed_set_path=args.closed_set,
        features_path=args.features,
        quality_path=args.quality,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()