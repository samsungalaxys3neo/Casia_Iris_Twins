from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PAIRS = PROJECT_ROOT / "data" / "features" / "daugman_strict_v3" / "pairs_features.csv.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "matching" / "daugman_strict_v3"


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


def read_pairs(input_pairs: Path) -> pd.DataFrame:
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

    return pd.read_csv(input_pairs, dtype=dtype_map)


def load_code_npz(path: Path):
    data = np.load(str(path))
    code = data["code"].astype(bool)
    mask = data["mask"].astype(bool)
    return code, mask


def masked_hamming_distance(code1, mask1, code2, mask2, shifts, min_common_bits=1000):
    """
    Calcola la masked Hamming Distance provando più shift angolari.

    code shape:
    [num_filters, radial_res, angular_res]

    Lo shift viene applicato sull'asse angolare, cioè axis=2.
    """

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
            best_hd = hd
            best_shift = int(shift)
            best_common_bits = common_bits

    if best_hd is None:
        return {
            "match_ok": False,
            "hamming_distance": np.nan,
            "best_shift": np.nan,
            "common_bits": best_common_bits,
            "error": "not enough common bits",
        }

    return {
        "match_ok": True,
        "hamming_distance": float(best_hd),
        "best_shift": best_shift,
        "common_bits": best_common_bits,
        "error": "",
    }


def summarize_by_relation(df: pd.DataFrame) -> dict:
    summary = {}

    for relation, group in df.groupby("relation"):
        values = pd.to_numeric(group["hamming_distance"], errors="coerce").dropna()

        if len(values) == 0:
            continue

        summary[str(relation)] = {
            "n": int(len(values)),
            "min": float(values.min()),
            "q1": float(values.quantile(0.25)),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "q3": float(values.quantile(0.75)),
            "max": float(values.max()),
            "std": float(values.std()),
        }

    return summary


def value_counts_dict(series: pd.Series) -> dict:
    counts = series.value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def run_matching(
    input_pairs: Path,
    output_dir: Path,
    max_shift: int,
    shift_step: int,
    min_common_bits: int,
):
    if not input_pairs.exists():
        raise FileNotFoundError(f"Pairs file non trovato: {input_pairs}")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_results = output_dir / "matching_results.csv.gz"
    output_summary = output_dir / "matching_summary.json"

    shifts = list(range(-max_shift, max_shift + 1, shift_step))

    print("=== GABOR CODE MATCHING ===")
    print(f"Pairs input:       {project_display_path(input_pairs)}")
    print(f"Output dir:        {project_display_path(output_dir)}")
    print(f"Results output:    {project_display_path(output_results)}")
    print(f"Shifts:            {shifts}")
    print(f"Min common bits:   {min_common_bits}")
    print()

    pairs = read_pairs(input_pairs)

    print(f"Coppie da confrontare: {len(pairs)}")

    unique_paths = sorted(
        set(pairs["code_path_1"].astype(str).tolist())
        | set(pairs["code_path_2"].astype(str).tolist())
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

    for idx, row in pairs.iterrows():
        code_path_1 = str(row["code_path_1"])
        code_path_2 = str(row["code_path_2"])

        code1, mask1 = code_cache[code_path_1]
        code2, mask2 = code_cache[code_path_2]

        match_result = masked_hamming_distance(
            code1,
            mask1,
            code2,
            mask2,
            shifts=shifts,
            min_common_bits=min_common_bits,
        )

        out = row.to_dict()
        out.update(match_result)
        results.append(out)

        if (idx + 1) % 5000 == 0:
            print(f"Match calcolati {idx + 1}/{len(pairs)}")

    mdf = pd.DataFrame(results)
    mdf.to_csv(output_results, index=False, compression="gzip")

    ok_df = mdf[mdf["match_ok"] == True].copy()

    summary = {
        "input_pairs_path": project_display_path(input_pairs),
        "output_results_path": project_display_path(output_results),
        "input_pairs": int(len(pairs)),
        "match_ok": int(len(ok_df)),
        "match_failed": int(len(pairs) - len(ok_df)),
        "max_shift": int(max_shift),
        "shift_step": int(shift_step),
        "shifts": [int(s) for s in shifts],
        "min_common_bits": int(min_common_bits),
        "relation_counts_input": value_counts_dict(pairs["relation"]),
        "relation_counts_match_ok": value_counts_dict(ok_df["relation"]) if len(ok_df) > 0 else {},
        "hamming_by_relation": summarize_by_relation(ok_df),
    }

    if len(ok_df) > 0:
        common_values = pd.to_numeric(ok_df["common_bits"], errors="coerce").dropna()
        summary["common_bits"] = {
            "min": int(common_values.min()),
            "q1": float(common_values.quantile(0.25)),
            "median": float(common_values.median()),
            "mean": float(common_values.mean()),
            "q3": float(common_values.quantile(0.75)),
            "max": int(common_values.max()),
        }

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== MATCHING SUMMARY ===")
    print(f"Input pairs:   {summary['input_pairs']}")
    print(f"Match OK:      {summary['match_ok']}")
    print(f"Match failed:  {summary['match_failed']}")
    print()

    print("Hamming Distance by relation:")
    for relation, stats in summary["hamming_by_relation"].items():
        print(
            f"{relation:30s} "
            f"n={stats['n']:6d} "
            f"mean={stats['mean']:.4f} "
            f"median={stats['median']:.4f} "
            f"q1={stats['q1']:.4f} "
            f"q3={stats['q3']:.4f}"
        )

    print()
    if "common_bits" in summary:
        print("Common bits:")
        for k, v in summary["common_bits"].items():
            print(f"  {k}: {v}")

    print()
    print(f"Results salvati in: {project_display_path(output_results)}")
    print(f"Summary salvato in: {project_display_path(output_summary)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-pairs",
        type=Path,
        default=DEFAULT_INPUT_PAIRS,
        help="Path a pairs_features.csv.gz. Default: data/features/daugman_strict_v3/pairs_features.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella output matching. Default: data/matching/daugman_strict_v3",
    )
    parser.add_argument(
        "--max-shift",
        type=int,
        default=16,
        help="Massimo shift angolare in colonne normalizzate.",
    )
    parser.add_argument(
        "--shift-step",
        type=int,
        default=2,
        help="Passo degli shift angolari.",
    )
    parser.add_argument(
        "--min-common-bits",
        type=int,
        default=1000,
        help="Minimo numero di bit comuni validi per accettare un matching.",
    )

    args = parser.parse_args()

    run_matching(
        input_pairs=resolve_from_project(args.input_pairs),
        output_dir=resolve_from_project(args.output_dir),
        max_shift=args.max_shift,
        shift_step=args.shift_step,
        min_common_bits=args.min_common_bits,
    )


if __name__ == "__main__":
    main()
