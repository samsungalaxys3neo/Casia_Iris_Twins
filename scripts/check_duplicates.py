from pathlib import Path
import pandas as pd


METADATA_PATH = Path("iris_twins_project/data/metadata/metadata.csv")
OUTPUT_PATH = Path("iris_twins_project/data/metadata/duplicate_report.csv")


def main():
    df = pd.read_csv(METADATA_PATH)

    duplicates = df[df["sha256"].duplicated(keep=False)].copy()
    duplicates = duplicates.sort_values(["sha256", "family_id", "twin_id", "eye", "filename"])

    if duplicates.empty:
        print("Nessun duplicato esatto trovato.")
        return

    duplicates.to_csv(OUTPUT_PATH, index=False)

    print(f"Trovate {len(duplicates)} righe coinvolte in duplicati esatti.")
    print(f"Report salvato in: {OUTPUT_PATH}")
    print()

    cols = [
        "sha256",
        "family_id",
        "twin_id",
        "eye",
        "subject_id",
        "iris_id",
        "filename",
        "relative_path",
    ]

    print(duplicates[cols].to_string(index=False))

    print("\n=== Analisi per gruppo duplicato ===")
    for sha, group in duplicates.groupby("sha256"):
        print("\nHash:", sha)
        print("Numero copie:", len(group))
        print("iris_id coinvolti:", sorted(group["iris_id"].unique()))
        print("family_id coinvolti:", sorted(group["family_id"].astype(str).unique()))
        print(group[["relative_path", "iris_id"]].to_string(index=False))


if __name__ == "__main__":
    main()