from pathlib import Path
import argparse
import csv
import gzip
import json
import random
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "daugman_strict_v3" / "features_report.csv"

DEFAULT_OUTPUT_PAIRS = PROJECT_ROOT / "data" / "features" / "daugman_strict_v3" / "pairs_features.csv.gz"
DEFAULT_OUTPUT_SUMMARY = PROJECT_ROOT / "data" / "features" / "daugman_strict_v3" / "pairs_features_summary.json"


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


def read_features_report(input_features: Path) -> pd.DataFrame:
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

    return df


def get_relation(row_i, row_j):
    same_family = row_i["family_id"] == row_j["family_id"]
    same_subject = row_i["subject_id"] == row_j["subject_id"]
    same_iris = row_i["iris_id"] == row_j["iris_id"]
    same_eye = row_i["eye"] == row_j["eye"]

    if same_iris:
        return "genuine"

    if same_subject and not same_eye:
        return "same_subject_different_eye"

    if same_family and not same_subject and same_eye:
        return "twin_same_eye"

    if same_family and not same_subject and not same_eye:
        return "twin_cross_eye"

    return "unrelated"


def run_build_pairs(
    input_features: Path,
    output_pairs: Path,
    output_summary: Path,
    max_unrelated: int,
    seed: int,
):
    if not input_features.exists():
        raise FileNotFoundError(f"Feature report non trovato: {input_features}")

    random.seed(seed)

    df = read_features_report(input_features)

    df = df[df["feature_ok"] == True].copy().reset_index(drop=True)
    df["sample_id"] = range(len(df))

    rows = df.to_dict("records")
    n = len(rows)

    print("=== BUILD FEATURE PAIRS ===")
    print(f"Input features:         {project_display_path(input_features)}")
    print(f"Input feature OK:       {n}")
    print(f"Max unrelated sampled:  {max_unrelated}")
    print(f"Seed:                   {seed}")
    print()

    selected_pairs = []
    unrelated_reservoir = []

    relation_counter_all = Counter()
    relation_counter_selected = Counter()

    unrelated_seen = 0

    for i in range(n):
        row_i = rows[i]

        for j in range(i + 1, n):
            row_j = rows[j]
            relation = get_relation(row_i, row_j)

            relation_counter_all[relation] += 1

            pair = {
                "sample_id_1": row_i["sample_id"],
                "sample_id_2": row_j["sample_id"],

                "image_1": row_i["relative_path"],
                "image_2": row_j["relative_path"],

                "code_path_1": row_i["code_path"],
                "code_path_2": row_j["code_path"],

                "family_id_1": row_i["family_id"],
                "family_id_2": row_j["family_id"],

                "subject_id_1": row_i["subject_id"],
                "subject_id_2": row_j["subject_id"],

                "iris_id_1": row_i["iris_id"],
                "iris_id_2": row_j["iris_id"],

                "twin_id_1": row_i["twin_id"],
                "twin_id_2": row_j["twin_id"],

                "eye_1": row_i["eye"],
                "eye_2": row_j["eye"],

                "relation": relation,
                "is_genuine": int(relation == "genuine"),
                "is_twin": int(relation in {"twin_same_eye", "twin_cross_eye"}),
                "is_unrelated": int(relation == "unrelated"),
            }

            if relation == "unrelated":
                unrelated_seen += 1

                # Reservoir sampling per non salvare milioni di coppie unrelated.
                if len(unrelated_reservoir) < max_unrelated:
                    unrelated_reservoir.append(pair)
                else:
                    k = random.randint(0, unrelated_seen - 1)
                    if k < max_unrelated:
                        unrelated_reservoir[k] = pair
            else:
                selected_pairs.append(pair)

        if (i + 1) % 250 == 0:
            print(f"Processati {i + 1}/{n} campioni")

    selected_pairs.extend(unrelated_reservoir)

    for pair in selected_pairs:
        relation_counter_selected[pair["relation"]] += 1

    for pair_id, pair in enumerate(selected_pairs):
        pair["pair_id"] = pair_id

    output_pairs.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pair_id",
        "sample_id_1",
        "sample_id_2",
        "image_1",
        "image_2",
        "code_path_1",
        "code_path_2",
        "family_id_1",
        "family_id_2",
        "subject_id_1",
        "subject_id_2",
        "iris_id_1",
        "iris_id_2",
        "twin_id_1",
        "twin_id_2",
        "eye_1",
        "eye_2",
        "relation",
        "is_genuine",
        "is_twin",
        "is_unrelated",
    ]

    with gzip.open(output_pairs, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_pairs)

    summary = {
        "input_features": project_display_path(input_features),
        "n_feature_images": int(n),
        "total_possible_pairs": int(n * (n - 1) // 2),
        "all_relation_counts": {k: int(v) for k, v in relation_counter_all.items()},
        "selected_relation_counts": {k: int(v) for k, v in relation_counter_selected.items()},
        "selected_pairs": int(len(selected_pairs)),
        "max_unrelated": int(max_unrelated),
        "seed": int(seed),
        "output_pairs": project_display_path(output_pairs),
    }

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== PAIRS FEATURE SUMMARY ===")
    print(f"Coppie possibili totali: {summary['total_possible_pairs']}")
    print(f"Coppie selezionate:      {len(selected_pairs)}")
    print()

    print("Tutte le coppie possibili:")
    for relation, count in relation_counter_all.most_common():
        print(f"  {relation:30s} {count}")

    print()
    print("Coppie selezionate:")
    for relation, count in relation_counter_selected.most_common():
        print(f"  {relation:30s} {count}")

    print()
    print(f"Salvato pairs in:   {project_display_path(output_pairs)}")
    print(f"Salvato summary in: {project_display_path(output_summary)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-features",
        type=Path,
        default=DEFAULT_INPUT_FEATURES,
        help="Path a features_report.csv. Default: data/features/daugman_strict_v3/features_report.csv",
    )
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=DEFAULT_OUTPUT_PAIRS,
        help="Path output pairs_features.csv.gz.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help="Path output pairs_features_summary.json.",
    )
    parser.add_argument(
        "--max-unrelated",
        type=int,
        default=50000,
        help="Numero massimo di coppie unrelated da campionare.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed random per il campionamento.",
    )

    args = parser.parse_args()

    run_build_pairs(
        input_features=resolve_from_project(args.input_features),
        output_pairs=resolve_from_project(args.output_pairs),
        output_summary=resolve_from_project(args.output_summary),
        max_unrelated=args.max_unrelated,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
