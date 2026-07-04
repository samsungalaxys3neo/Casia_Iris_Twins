#!/usr/bin/env python3
"""
Run the final iris-twins biometric pipeline.

Usage:
    python scripts/run_pipeline.py --raw-root /absolute/path/to/CASIA-Iris-Twins

The raw dataset can be anywhere on the user's machine. The only requirement is
that --raw-root points to the folder containing CASIA family folders such as:
    00/1L, 00/1R, 00/2L, 00/2R, ..., 99/...

This runner intentionally avoids computer-specific paths and does not require
renaming the repository folder to iris_twins_project.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPECTED_SUBFOLDERS = {"1L", "1R", "2L", "2R"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the iris twins pipeline.")
    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
        help="Path to the raw CASIA-Iris-Twins dataset folder.",
    )
    parser.add_argument(
        "--max-unrelated",
        type=int,
        default=50000,
        help="Number of unrelated feature pairs to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for unrelated pair sampling.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Default: the current interpreter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Start from a stage name.",
    )
    parser.add_argument(
        "--stop-after",
        default=None,
        help="Stop after a stage name.",
    )
    return parser.parse_args()


def validate_dataset_root(raw_root: Path) -> Path:
    """Validate the external dataset root without assuming any fixed location."""
    raw_root = raw_root.expanduser().resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_root}")
    if not raw_root.is_dir():
        raise NotADirectoryError(f"--raw-root must be a directory: {raw_root}")

    family_dirs = sorted(
        p for p in raw_root.iterdir() if p.is_dir() and re_fullmatch_two_digits(p.name)
    )
    if not family_dirs:
        raise FileNotFoundError(
            "The dataset root does not look like CASIA-Iris-Twins.\n"
            f"Given path: {raw_root}\n"
            "Expected folders like 00, 01, 02, ... containing 1L/1R/2L/2R.\n"
            "Pass the extracted dataset folder with --raw-root, not the code repository."
        )

    has_expected_subfolder = False
    for family_dir in family_dirs[:10]:
        subfolders = {p.name for p in family_dir.iterdir() if p.is_dir()}
        if EXPECTED_SUBFOLDERS.intersection(subfolders):
            has_expected_subfolder = True
            break

    if not has_expected_subfolder:
        raise FileNotFoundError(
            "CASIA family folders were found, but no 1L/1R/2L/2R subfolders were found.\n"
            f"Given path: {raw_root}\n"
            "Check whether the dataset is nested one level deeper and pass that inner folder."
        )

    return raw_root


def re_fullmatch_two_digits(value: str) -> bool:
    return len(value) == 2 and value.isdigit()


def run_step(
    name: str,
    command: list[str],
    cwd: Path,
    dry_run: bool = False,
) -> None:
    print("\n" + "=" * 80)
    print(f"[{name}]")
    print(" ".join(command))
    print("=" * 80)
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def selected_steps(
    steps: list[tuple[str, list[str]]],
    start_at: str | None,
    stop_after: str | None,
) -> list[tuple[str, list[str]]]:
    names = [name for name, _ in steps]

    if start_at is not None and start_at not in names:
        raise ValueError(f"Unknown --start-at stage: {start_at}\nAvailable: {names}")
    if stop_after is not None and stop_after not in names:
        raise ValueError(f"Unknown --stop-after stage: {stop_after}\nAvailable: {names}")

    start_idx = names.index(start_at) if start_at else 0
    stop_idx = names.index(stop_after) if stop_after else len(steps) - 1

    if start_idx > stop_idx:
        raise ValueError("--start-at cannot be after --stop-after")

    return steps[start_idx : stop_idx + 1]


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    script_dir = repo_root / "scripts"
    run_cwd = repo_root
    raw_root = validate_dataset_root(args.raw_root)
    py = args.python

    steps: list[tuple[str, list[str]]] = [
        (
            "build_metadata",
            [
                py,
                str(script_dir / "build_metadata.py"),
                "--root",
                str(raw_root),
                "--output",
                "data/metadata",
            ],
        ),
        ("check_duplicates", [py, str(script_dir / "check_duplicates.py")]),
        ("create_clean_metadata", [py, str(script_dir / "create_clean_metadata.py")]),
        ("build_pairs", [py, str(script_dir / "build_pairs.py")]),
        (
            "quality_audit",
            [
                py,
                str(script_dir / "quality_audit.py"),
                "--dataset-root",
                str(raw_root),
            ],
        ),
        (
            "inspect_quality_outliers",
            [
                py,
                str(script_dir / "inspect_quality_outliers.py"),
                "--dataset-root",
                str(raw_root),
            ],
        ),
        ("create_final_metadata", [py, str(script_dir / "create_final_metadata.py")]),
        (
            "segmentation_daugman_style",
            [
                py,
                str(script_dir / "segmentation_daugman_style.py"),
                "--dataset-root",
                str(raw_root),
            ],
        ),
        ("filter_daugman_strict", [py, str(script_dir / "filter_daugman_strict.py")]),
        ("audit_strict_metadata", [py, str(script_dir / "audit_strict_metadata.py")]),
        (
            "create_strict_subsets",
            [
                py,
                str(script_dir / "create_strict_subsets.py"),
                "--input",
                "data/metadata/metadata_daugman_strict.csv",
                "--output-min2",
                "data/metadata/metadata_daugman_strict_min2.csv",
                "--output-complete",
                "data/metadata/metadata_daugman_strict_complete_families.csv",
                "--output-summary",
                "data/metadata/strict_subsets_summary.txt",
            ],
        ),
        (
            "normalize_daugman_strict_v3",
            [
                py,
                str(script_dir / "normalize_daugman_strict_v3.py"),
                "--dataset-root",
                str(raw_root),
            ],
        ),
        ("extract_gabor_features", [py, str(script_dir / "extract_gabor_features.py")]),
        (
            "build_feature_pairs",
            [
                py,
                str(script_dir / "build_feature_pairs.py"),
                "--max-unrelated",
                str(args.max_unrelated),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "match_gabor_codes",
            [
                py,
                str(script_dir / "match_gabor_codes.py"),
                "--max-shift",
                "16",
                "--shift-step",
                "2",
                "--min-common-bits",
                "1000",
            ],
        ),
        ("analyze_matching_results", [py, str(script_dir / "analyze_matching_results.py")]),
        ("bootstrap_permutation_tests", [py, str(script_dir / "bootstrap_permutation_tests.py")]),
        ("extract_lbp_blob_features", [py, str(script_dir / "extract_lbp_blob_features.py")]),
        ("match_lbp_blob_features", [py, str(script_dir / "match_lbp_blob_features.py")]),
        ("closed_set_identification", [py, str(script_dir / "closed_set_identification.py")]),
        ("failure_analysis", [py, str(script_dir / "failure_analysis.py")]),
        (
            "compare_closed_set_success_failure",
            [py, str(script_dir / "compare_closed_set_success_failure.py")],
        ),
        ("quality_vs_rank1_errors", [py, str(script_dir / "quality_vs_rank1_errors.py")]),
    ]

    steps_to_run = selected_steps(steps, args.start_at, args.stop_after)

    print(f"Repository root: {repo_root}")
    print(f"Repository folder name: {repo_root.name}")
    print(f"Execution cwd: {run_cwd}")
    print(f"Raw dataset: {raw_root}")
    print(f"Number of stages: {len(steps_to_run)}")

    for name, command in steps_to_run:
        run_step(name, command, cwd=run_cwd, dry_run=args.dry_run)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
