from pathlib import Path
import argparse
import csv
import gzip
import json
from collections import Counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_METADATA = PROJECT_ROOT / "data" / "metadata" / "metadata_clean.csv"
DEFAULT_OUTPUT_PAIRS = PROJECT_ROOT / "data" / "metadata" / "pairs.csv.gz"
DEFAULT_OUTPUT_SUMMARY = PROJECT_ROOT / "data" / "metadata" / "pairs_summary.json"


def resolve_from_project(path: Path) -> Path:
    """
    Resolve a path in a portable way.
    If the path is relative, interpret it relative to the project root.
    If the path is absolute, keep it as it is.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_display_path(path: Path) -> str:
    """
    Return a clean path for terminal output.
    If the path is inside the project, show it relative to PROJECT_ROOT.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return "<external_path>"


def read_metadata(input_metadata: Path) -> pd.DataFrame:
    """
    Read metadata while preserving important identifier columns as strings.
    """
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
    }
    return pd.read_csv(input_metadata, dtype=dtype_map)


def get_relation(row_i, row_j):
    """
    Classifica la relazione tra due immagini.

    genuine:
        stessa iride (immagini diverse)

    same_subject_different_eye:
        stesso soggetto, occhio diverso

    twin_same_eye:
        stessa famiglia, gemello diverso, stesso occhio

    twin_cross_eye:
        stessa famiglia, gemello diverso, occhio diverso

    unrelated:
        famiglie diverse
    """

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


def build_pairs(input_metadata: Path, output_pairs: Path, output_summary: Path):
    if not input_metadata.exists():
        raise FileNotFoundError(
            f"Non trovo {input_metadata}. Prima crea metadata_clean.csv."
        )

    df = read_metadata(input_metadata)

    sort_cols = ["family_id", "twin_id", "eye", "image_idx", "filename"]
    missing_sort_cols = [col for col in sort_cols if col not in df.columns]
    if missing_sort_cols:
        raise ValueError(f"Colonne mancanti per ordinamento: {missing_sort_cols}")

    # Ordine stabile
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ID numerico comodo per le matrici di distanza successive
    df["sample_id"] = range(len(df))

    n = len(df)
    total_pairs = n * (n - 1) // 2

    print("=== BUILD PAIRS ===")
    print(f"Immagini in metadata_clean: {n}")
    print(f"Coppie totali All-vs-All attese: {total_pairs}")
    print(f"Output pairs: {project_display_path(output_pairs)}")
    print(f"Output summary: {project_display_path(output_summary)}")
    print()

    output_pairs.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    relation_counter = Counter()
    family_counter = Counter()
    iris_pair_counter = Counter()

    fieldnames = [
        "pair_id",
        "sample_id_1",
        "sample_id_2",
        "image_1",
        "image_2",
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
        "same_subject",
        "same_eye",
    ]

    rows = df.to_dict("records")

    with gzip.open(output_pairs, mode="wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        pair_id = 0

        for i in range(n):
            row_i = rows[i]

            for j in range(i + 1, n):
                row_j = rows[j]

                relation = get_relation(row_i, row_j)

                is_genuine = relation == "genuine"
                is_twin = relation in {"twin_same_eye", "twin_cross_eye"}
                is_unrelated = relation == "unrelated"
                same_subject = row_i["subject_id"] == row_j["subject_id"]
                same_eye = row_i["eye"] == row_j["eye"]

                writer.writerow({
                    "pair_id": pair_id,
                    "sample_id_1": row_i["sample_id"],
                    "sample_id_2": row_j["sample_id"],
                    "image_1": row_i["relative_path"],
                    "image_2": row_j["relative_path"],
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
                    "is_genuine": int(is_genuine),
                    "is_twin": int(is_twin),
                    "is_unrelated": int(is_unrelated),
                    "same_subject": int(same_subject),
                    "same_eye": int(same_eye),
                })

                relation_counter[relation] += 1

                if row_i["family_id"] == row_j["family_id"]:
                    family_counter[row_i["family_id"]] += 1

                iris_pair_key = tuple(sorted([row_i["iris_id"], row_j["iris_id"]]))
                iris_pair_counter[iris_pair_key] += 1

                pair_id += 1

            if (i + 1) % 250 == 0:
                print(f"Processate righe: {i + 1}/{n} | coppie scritte: {pair_id}")

    summary = {
        "n_images": n,
        "total_pairs": total_pairs,
        "written_pairs": pair_id,
        "relation_counts": dict(relation_counter),
        "family_internal_pair_counts": dict(family_counter),
    }

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== PAIRS SUMMARY ===")
    print(f"Coppie scritte: {pair_id}")
    print()

    for relation, count in relation_counter.most_common():
        percentage = 100 * count / pair_id
        print(f"{relation:30s} {count:10d}  ({percentage:6.3f}%)")

    print()
    print(f"Salvato file coppie in: {project_display_path(output_pairs)}")
    print(f"Salvato summary in: {project_display_path(output_summary)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_METADATA,
        help="Path a metadata_clean.csv. Default: data/metadata/metadata_clean.csv",
    )
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=DEFAULT_OUTPUT_PAIRS,
        help="Path dove salvare pairs.csv.gz. Default: data/metadata/pairs.csv.gz",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help="Path dove salvare pairs_summary.json. Default: data/metadata/pairs_summary.json",
    )

    args = parser.parse_args()

    input_metadata = resolve_from_project(args.input)
    output_pairs = resolve_from_project(args.output_pairs)
    output_summary = resolve_from_project(args.output_summary)

    build_pairs(input_metadata, output_pairs, output_summary)


if __name__ == "__main__":
    main()