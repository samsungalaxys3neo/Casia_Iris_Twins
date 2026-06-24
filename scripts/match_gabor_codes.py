from pathlib import Path
import argparse
import gzip
import json
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PAIRS = PROJECT_ROOT / "data/features/daugman_strict_v3/pairs_features.csv.gz"

OUTPUT_DIR = PROJECT_ROOT / "data/matching/daugman_strict_v3"
OUTPUT_RESULTS = OUTPUT_DIR / "matching_results.csv.gz"
OUTPUT_SUMMARY = OUTPUT_DIR / "matching_summary.json"


def load_code_npz(path):
    data = np.load(path)
    code = data["code"].astype(bool)
    mask = data["mask"].astype(bool)
    return code, mask


def masked_hamming_distance(code1, mask1, code2, mask2, shifts, min_common_bits=1000):
    """
    Calcola la masked Hamming Distance provando più shift angolari.
    Axis:
    code shape = [num_filters, radial_res, angular_res]
    shift sull'asse angular_res, cioè axis=2.
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


def summarize_by_relation(df):
    summary = {}

    for relation, group in df.groupby("relation"):
        values = pd.to_numeric(group["hamming_distance"], errors="coerce").dropna()

        if len(values) == 0:
            continue

        summary[relation] = {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-shift",
        type=int,
        default=16,
        help="Massimo shift angolare in colonne normalizzate",
    )
    parser.add_argument(
        "--shift-step",
        type=int,
        default=2,
        help="Passo degli shift angolari",
    )
    parser.add_argument(
        "--min-common-bits",
        type=int,
        default=1000,
        help="Minimo numero di bit comuni validi per accettare un matching",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shifts = list(range(-args.max_shift, args.max_shift + 1, args.shift_step))

    print("=== GABOR CODE MATCHING ===")
    print(f"Pairs input:       {INPUT_PAIRS}")
    print(f"Output:            {OUTPUT_RESULTS}")
    print(f"Shifts:            {shifts}")
    print(f"Min common bits:   {args.min_common_bits}")
    print()

    pairs = pd.read_csv(
        INPUT_PAIRS,
        dtype={
            "family_id_1": str,
            "family_id_2": str,
            "twin_id_1": str,
            "twin_id_2": str,
            "eye_1": str,
            "eye_2": str,
        }
    )

    print(f"Coppie da confrontare: {len(pairs)}")

    # Precarichiamo tutti i codici usati nel matching.
    unique_paths = sorted(
        set(pairs["code_path_1"].tolist()) | set(pairs["code_path_2"].tolist())
    )

    print(f"Codici unici da caricare: {len(unique_paths)}")

    code_cache = {}

    for idx, path in enumerate(unique_paths):
        code_cache[path] = load_code_npz(path)

        if (idx + 1) % 250 == 0:
            print(f"Caricati {idx + 1}/{len(unique_paths)} codici")

    results = []

    for idx, row in pairs.iterrows():
        code1, mask1 = code_cache[row["code_path_1"]]
        code2, mask2 = code_cache[row["code_path_2"]]

        match_result = masked_hamming_distance(
            code1,
            mask1,
            code2,
            mask2,
            shifts=shifts,
            min_common_bits=args.min_common_bits,
        )

        out = row.to_dict()
        out.update(match_result)
        results.append(out)

        if (idx + 1) % 5000 == 0:
            print(f"Match calcolati {idx + 1}/{len(pairs)}")

    mdf = pd.DataFrame(results)
    mdf.to_csv(OUTPUT_RESULTS, index=False, compression="gzip")

    ok_df = mdf[mdf["match_ok"] == True].copy()

    summary = {
        "input_pairs": int(len(pairs)),
        "match_ok": int(len(ok_df)),
        "match_failed": int(len(pairs) - len(ok_df)),
        "max_shift": args.max_shift,
        "shift_step": args.shift_step,
        "shifts": shifts,
        "min_common_bits": args.min_common_bits,
        "relation_counts_input": pairs["relation"].value_counts().to_dict(),
        "relation_counts_match_ok": ok_df["relation"].value_counts().to_dict(),
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

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
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
    print(f"Results salvati in: {OUTPUT_RESULTS}")
    print(f"Summary salvato in: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()