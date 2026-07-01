from pathlib import Path
import argparse
import hashlib
import re
import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_FAMILIES = {f"{i:02d}" for i in range(100)}
EXPECTED_SUBFOLDERS = {"1L", "1R", "2L", "2R"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "CASIA-Iris-Twins"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata"

# funzioni per il path, in modo da non dipendere dalla cwd del terminale, ma sempre dalla root del progetto 
def resolve_from_project(path: Path) -> Path:
    """
    Resolve a path. If it is relative, interpret it relative to the project root,
    not relative to the terminal current working directory.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_display_path(path: Path) -> str:
    """
    Return a clean path for reports/prints.
    If possible, show it relative to the project root.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return "<external_dataset_root>"

# Esempio nome atteso CASIA Twins: S3XXYENN.jpg
# XX = family, Y = twin_id, E = L/R, NN = image index
CASIA_TWINS_RE = re.compile(
    r"^S3(?P<family>\d{2})(?P<twin>[12])(?P<eye>[LR])(?P<image_idx>\d+)$",
    re.IGNORECASE
)


def sha256_file(path: Path, chunk_size: int = 8192) -> str:
    """Calcola hash del file, utile per trovare duplicati esatti."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_image_info(path: Path) -> dict:
    """Legge dimensioni e modo immagine. Se fallisce, segna errore."""
    try:
        with Image.open(path) as img:
            return {
                "readable": True,
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "error": "",
            }
    except Exception as e:
        return {
            "readable": False,
            "width": None,
            "height": None,
            "mode": None,
            "format": None,
            "error": str(e),
        }


def parse_filename(path: Path) -> dict:
    """Prova a estrarre metadati dal nome file, se rispetta lo schema CASIA."""
    stem = path.stem
    match = CASIA_TWINS_RE.match(stem)

    if not match:
        return {
            "filename_family_id": None,
            "filename_twin_id": None,
            "filename_eye": None,
            "filename_image_idx": None,
            "filename_pattern_ok": False,
        }

    gd = match.groupdict()
    return {
        "filename_family_id": gd["family"],
        "filename_twin_id": gd["twin"],
        "filename_eye": gd["eye"].upper(),
        "filename_image_idx": gd["image_idx"],
        "filename_pattern_ok": True,
    }


def build_metadata(root: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    audit_warnings = []

    if not root.exists():
        raise FileNotFoundError(f"Dataset root non trovato: {root}")

    family_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name
    )

    found_families = {p.name for p in family_dirs}
    missing_families = sorted(EXPECTED_FAMILIES - found_families)
    extra_families = sorted(found_families - EXPECTED_FAMILIES)

    if missing_families:
        audit_warnings.append(f"Famiglie mancanti: {missing_families}")
    if extra_families:
        audit_warnings.append(f"Cartelle famiglia extra/non attese: {extra_families}")

    for family_dir in family_dirs:
        family_id = family_dir.name

        if family_id not in EXPECTED_FAMILIES:
            audit_warnings.append(f"Cartella famiglia non standard: {family_dir}")

        subfolders = {p.name for p in family_dir.iterdir() if p.is_dir()}
        missing_subfolders = sorted(EXPECTED_SUBFOLDERS - subfolders)
        extra_subfolders = sorted(subfolders - EXPECTED_SUBFOLDERS)

        if missing_subfolders:
            audit_warnings.append(
                f"Famiglia {family_id}: sottocartelle mancanti {missing_subfolders}"
            )
        if extra_subfolders:
            audit_warnings.append(
                f"Famiglia {family_id}: sottocartelle extra {extra_subfolders}"
            )

        for subfolder_name in sorted(EXPECTED_SUBFOLDERS):
            subfolder = family_dir / subfolder_name

            if not subfolder.exists():
                continue

            twin_id = subfolder_name[0]   # "1" oppure "2"
            eye = subfolder_name[1]       # "L" oppure "R"

            subject_id = f"F{family_id}_T{twin_id}"
            iris_id = f"{subject_id}_{eye}"

            image_paths = sorted(
                [
                    p for p in subfolder.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
            )

            for local_idx, img_path in enumerate(image_paths, start=1):
                filename_info = parse_filename(img_path)
                img_info = read_image_info(img_path)

                file_hash = None
                file_size = None

                try:
                    file_size = img_path.stat().st_size
                    file_hash = sha256_file(img_path)
                except Exception as e:
                    img_info["readable"] = False
                    img_info["error"] = f"{img_info['error']} | hash/stat error: {e}"

                filename_image_idx = filename_info["filename_image_idx"]
                image_idx = filename_image_idx if filename_image_idx is not None else str(local_idx).zfill(2)

                folder_filename_consistent = True

                if filename_info["filename_pattern_ok"]:
                    folder_filename_consistent = (
                        filename_info["filename_family_id"] == family_id
                        and filename_info["filename_twin_id"] == twin_id
                        and filename_info["filename_eye"] == eye
                    )

                rows.append({
                    "family_id": family_id,
                    "twin_id": twin_id,
                    "eye": eye,
                    "subject_id": subject_id,
                    "iris_id": iris_id,
                    "image_idx": image_idx,

                    "filename": img_path.name,
                    "relative_path": str(img_path.relative_to(root)),

                    "file_extension": img_path.suffix.lower(),
                    "file_size_bytes": file_size,
                    "sha256": file_hash,

                    "width": img_info["width"],
                    "height": img_info["height"],
                    "mode": img_info["mode"],
                    "format": img_info["format"],
                    "readable": img_info["readable"],
                    "error": img_info["error"],

                    "filename_pattern_ok": filename_info["filename_pattern_ok"],
                    "filename_family_id": filename_info["filename_family_id"],
                    "filename_twin_id": filename_info["filename_twin_id"],
                    "filename_eye": filename_info["filename_eye"],
                    "filename_image_idx": filename_info["filename_image_idx"],
                    "folder_filename_consistent": folder_filename_consistent,
                })

    df = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)

    # Report conteggi principali
    report_lines = []
    report_lines.append("=== DATASET AUDIT REPORT ===")
    report_lines.append(f"Dataset root: {root.resolve()}")
    report_lines.append(f"Famiglie attese: {len(EXPECTED_FAMILIES)}")
    report_lines.append(f"Famiglie trovate: {len(found_families)}")
    report_lines.append(f"Immagini totali trovate: {len(df)}")

    if len(df) > 0:
        report_lines.append("")
        report_lines.append("=== Conteggio per family_id ===")
        report_lines.append(df.groupby("family_id").size().to_string())

        report_lines.append("")
        report_lines.append("=== Conteggio per sottoclasse 1L/1R/2L/2R ===")
        df["folder_class"] = df["twin_id"] + df["eye"]
        report_lines.append(df.groupby("folder_class").size().to_string())

        report_lines.append("")
        report_lines.append("=== Immagini non leggibili ===")
        unreadable = df[df["readable"] == False]
        report_lines.append(str(len(unreadable)))

        report_lines.append("")
        report_lines.append("=== Filename non conformi allo schema CASIA ===")
        bad_pattern = df[df["filename_pattern_ok"] == False]
        report_lines.append(str(len(bad_pattern)))

        report_lines.append("")
        report_lines.append("=== Incoerenze cartella/nome file ===")
        inconsistent = df[df["folder_filename_consistent"] == False]
        report_lines.append(str(len(inconsistent)))

        report_lines.append("")
        report_lines.append("=== Duplicati esatti, stesso hash ===")
        duplicated_hashes = df[df["sha256"].duplicated(keep=False) & df["sha256"].notna()]
        report_lines.append(str(len(duplicated_hashes)))

    report_lines.append("")
    report_lines.append("=== Warnings ===")
    if audit_warnings:
        report_lines.extend(audit_warnings)
    else:
        report_lines.append("Nessun warning strutturale.")

    report_path = output_dir / "audit_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nSalvato metadata in: {metadata_path}")
    print(f"Salvato report in: {report_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path alla cartella CASIA-Iris-Twins"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=PROJECT_ROOT / "data" / "metadata",
        help="Cartella dove salvare metadata.csv e audit_report.txt"
    )

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output)

    build_metadata(root, output_dir)


if __name__ == "__main__":
    main()

