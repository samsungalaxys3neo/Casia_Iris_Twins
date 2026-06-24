from pathlib import Path
import csv
import gzip
import json
from collections import Counter, defaultdict

import pandas as pd


INPUT_METADATA = Path("iris_twins_project/data/metadata/metadata_clean.csv")
OUTPUT_PAIRS = Path("iris_twins_project/data/metadata/pairs.csv.gz")
OUTPUT_SUMMARY = Path("iris_twins_project/data/metadata/pairs_summary.json")


def get_relation(row_i, row_j):
    """
    Classifica la relazione tra due immagini.

    genuine:
        stesso soggetto + stesso occhio, ma immagini diverse

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


def main():
    if not INPUT_METADATA.exists():
        raise FileNotFoundError(
            f"Non trovo {INPUT_METADATA}. Prima crea metadata_clean.csv."
        )

    df = pd.read_csv(INPUT_METADATA)

    # Ordine stabile
    df = df.sort_values(
        ["family_id", "twin_id", "eye", "image_idx", "filename"]
    ).reset_index(drop=True)

    # ID numerico comodo per le matrici di distanza successive
    df["sample_id"] = range(len(df))

    n = len(df)
    total_pairs = n * (n - 1) // 2

    print("=== BUILD PAIRS ===")
    print(f"Immagini in metadata_clean: {n}")
    print(f"Coppie totali All-vs-All attese: {total_pairs}")
    print(f"Output: {OUTPUT_PAIRS}")
    print()

    OUTPUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)

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

    with gzip.open(OUTPUT_PAIRS, mode="wt", newline="", encoding="utf-8") as f:
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

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== PAIRS SUMMARY ===")
    print(f"Coppie scritte: {pair_id}")
    print()

    for relation, count in relation_counter.most_common():
        percentage = 100 * count / pair_id
        print(f"{relation:30s} {count:10d}  ({percentage:6.3f}%)")

    print()
    print(f"Salvato file coppie in: {OUTPUT_PAIRS}")
    print(f"Salvato summary in: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()