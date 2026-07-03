#!/usr/bin/env python3
"""
DEMO ORALE - STRICT SAMPLE - V3 ROBUSTA

Fix principale rispetto alle versioni precedenti:
- la selezione del sample viene fatta SOLO tra immagini strict che hanno un overlay
  di segmentazione corrispondente al proprio stem/filename;
- un overlay viene accettato SOLO se il suo nome contiene lo stem del sample;
- lo stesso overlay non può essere riusato per più immagini;
- se qualcosa non viene trovato, viene segnato come missing, NON sostituito con una
  foto sbagliata.

Run:
    cd path/Casia_Iris_Twins
    python3 scripts/demo_orale_strict_sample_v3.py --seed 42 --families 2 --images-per-iris 2 --open

Output:
    demo_orale_strict_sample_v3/
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


DEFAULT_REPO_ROOT = Path("/Users/sarahoualli/Desktop/BIO/iris_twins_project")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ARRAY_EXT = {".npy", ".npz"}
EYE_FOLDERS = {"1L", "1R", "2L", "2R"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a robust oral demo from strict final outputs.")
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--families", type=int, default=2)
    p.add_argument("--images-per-iris", type=int, default=2)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--open", action="store_true")
    p.add_argument("--no-clean", action="store_true")
    p.add_argument(
        "--allow-missing-overlays",
        action="store_true",
        help="Allow selecting strict images even if a true segmentation overlay is missing. Default: false.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def reset_dir(path: Path, clean: bool = True) -> None:
    if path.exists() and clean:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"Missing one of {candidates}. Available columns: {list(df.columns)}")
    return None


def try_font(size: int):
    for name in ["Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def list_files(root: Path, exts: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def resolve_path(value: str | Path, bases: list[Path]) -> Path | None:
    if value is None:
        return None
    p = Path(str(value))
    candidates = [p] if p.is_absolute() else [b / p for b in bases]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return None


def search_raw_by_stem(raw_root: Path, stem: str) -> Path | None:
    stem_l = stem.lower()
    for p in list_files(raw_root, IMAGE_EXT):
        if stem_l in p.name.lower():
            return p.resolve()
    return None


def build_exact_stem_index(files: list[Path]) -> dict[str, Path]:
    """Index files by stems appearing in their filename.

    Works with names such as:
        10__1L__S3101L09.jpg.png
        22__1L__S3221L05.jpg_mask.png
        22__1L__S3221L05.jpg.npz
    The extracted key is the token starting with S and containing L/R before digits.
    To stay robust, we also let exact lookup scan by selected stem later.
    """
    index: dict[str, Path] = {}
    for p in files:
        name = p.name
        parts = name.replace(".jpg", " ").replace(".png", " ").replace(".npz", " ").replace(".npy", " ").replace("__", " ").replace("_", " ").split()
        for token in parts:
            token = token.strip()
            if token.startswith("S") and any(x in token for x in ["L", "R"]):
                index.setdefault(token, p)
    return index


def exact_find_by_stem(files: list[Path], stem: str, used: set[Path] | None = None) -> Path | None:
    """Return a file only if the selected stem is literally in the file name.

    This is the critical anti-bug function: it never returns a generic overlay.
    """
    used = used or set()
    stem_l = stem.lower()
    matches = [p for p in files if p not in used and stem_l in p.name.lower()]
    if not matches:
        return None
    # Prefer shorter path/name and images for visual artifacts.
    matches.sort(key=lambda p: (len(str(p)), p.name))
    return matches[0]


def normalize_array_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    arr = arr.astype(float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if mx - mn < 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn)
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def load_array_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        key = "arr_0" if "arr_0" in z.files else z.files[0]
        return z[key]
    if path.suffix.lower() in IMAGE_EXT:
        return np.array(Image.open(path).convert("L"))
    raise ValueError(f"Unsupported file type: {path}")


def render_array_file(path: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = load_array_file(path)
    if arr.ndim == 2:
        img = normalize_array_to_uint8(arr)
    elif arr.ndim == 3:
        if arr.shape[0] <= 8:
            channels = [arr[i] for i in range(arr.shape[0])]
        elif arr.shape[-1] <= 8:
            channels = [arr[:, :, i] for i in range(arr.shape[-1])]
        else:
            channels = [arr[:, :, 0]]
        img = np.concatenate([normalize_array_to_uint8(ch) for ch in channels], axis=1)
    else:
        img = normalize_array_to_uint8(arr.reshape(arr.shape[0], -1))
    Image.fromarray(img).save(out_path)
    return out_path


def image_to_thumb(path: Path, w: int, h: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def save_contact_sheet(items: list[tuple[Path, str]], out_path: Path, title: str, cols: int = 3) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h, title_h, label_h = 430, 250, 58, 50

    if not items:
        sheet = Image.new("RGB", (1180, 320), "white")
        d = ImageDraw.Draw(sheet)
        d.text((20, 28), title, fill="black", font=try_font(24))
        d.text((20, 100), "Nessun file trovato per questi sample.", fill="red", font=try_font(18))
        d.text((20, 145), "Apri tables/artifact_lookup_log.csv per vedere i missing.", fill="black", font=try_font(16))
        sheet.save(out_path)
        return

    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, title_h + rows * (thumb_h + label_h)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((14, 18), title, fill="black", font=try_font(22))
    font_label = try_font(14)

    for i, (path, label) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * thumb_w, title_h + r * (thumb_h + label_h)
        try:
            thumb = image_to_thumb(path, thumb_w, thumb_h)
        except Exception as e:
            thumb = Image.new("RGB", (thumb_w, thumb_h), "white")
            ImageDraw.Draw(thumb).text((10, 10), f"Errore apertura:\n{path.name}\n{e}", fill="red", font=font_label)
        sheet.paste(thumb, (x, y))
        d.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill="white")
        d.text((x + 8, y + thumb_h + 10), label[:76], fill="black", font=font_label)

    sheet.save(out_path)


# ---------------------------------------------------------------------
# Metadata and sample selection
# ---------------------------------------------------------------------

def load_strict_metadata(repo_root: Path, strict_metadata: Path, overlay_files: list[Path]) -> pd.DataFrame:
    if not strict_metadata.exists():
        raise FileNotFoundError(f"Strict metadata not found: {strict_metadata}")

    df = pd.read_csv(strict_metadata)
    family_col = find_col(df, ["family_id", "family", "familyid"], required=True)
    image_col = find_col(df, ["image_path", "path", "filepath", "file_path", "raw_path", "clean_path", "original_path", "absolute_path", "img_path"])
    filename_col = find_col(df, ["filename", "file_name", "image_name", "name"])
    folder_col = find_col(df, ["folder_class", "class", "eye_folder", "folder"])
    twin_col = find_col(df, ["twin_id", "twin"])
    eye_col = find_col(df, ["eye", "eye_side"])
    subject_col = find_col(df, ["subject_id", "subject", "person_id"])
    iris_col = find_col(df, ["iris_id", "irisid"])

    rows = []
    for _, row in df.iterrows():
        if filename_col:
            filename = Path(safe_str(row[filename_col])).name
        elif image_col:
            filename = Path(safe_str(row[image_col])).name
        else:
            raise RuntimeError("No filename/image_path column in strict metadata.")
        stem = Path(filename).stem

        resolved = None
        if image_col:
            resolved = resolve_path(safe_str(row[image_col]), [repo_root, strict_metadata.parent, strict_metadata.parent.parent, strict_metadata.parent.parent.parent, repo_root / "data/raw"])
        if resolved is None:
            resolved = search_raw_by_stem(repo_root / "data/raw", stem)

        folder_class = safe_str(row[folder_col]).upper() if folder_col else ""
        if folder_class not in EYE_FOLDERS and twin_col and eye_col:
            twin = safe_str(row[twin_col]).replace(".0", "").strip()
            eye_tmp = safe_str(row[eye_col]).upper().strip()[:1]
            candidate = f"{twin}{eye_tmp}"
            if candidate in EYE_FOLDERS:
                folder_class = candidate
        if folder_class not in EYE_FOLDERS and resolved is not None:
            for part in resolved.parts:
                if part.upper() in EYE_FOLDERS:
                    folder_class = part.upper()
                    break
        if folder_class not in EYE_FOLDERS:
            folder_class = "UNK"

        family_id = safe_str(row[family_col])
        twin_id = safe_str(row[twin_col]) if twin_col else (folder_class[0] if folder_class != "UNK" else "")
        eye = safe_str(row[eye_col]).upper()[:1] if eye_col else (folder_class[1] if folder_class != "UNK" else "")
        subject_id = safe_str(row[subject_col]) if subject_col else f"{family_id}_{twin_id}"
        iris_id = safe_str(row[iris_col]) if iris_col else f"{subject_id}_{eye}"

        overlay = exact_find_by_stem(overlay_files, stem)

        rows.append({
            "family_id": family_id,
            "twin_id": twin_id,
            "eye": eye,
            "subject_id": subject_id,
            "iris_id": iris_id,
            "folder_class": folder_class,
            "filename": filename,
            "stem": stem,
            "source_path": str(resolved) if resolved else "",
            "segmentation_overlay_path": str(overlay) if overlay else "",
            "has_segmentation_overlay": bool(overlay),
        })

    return pd.DataFrame(rows)


def select_demo_sample(meta: pd.DataFrame, families: int, images_per_iris: int, seed: int, require_overlays: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool_meta = meta.copy()
    if require_overlays:
        pool_meta = pool_meta[pool_meta["has_segmentation_overlay"] == True].copy()
        if pool_meta.empty:
            raise RuntimeError("No strict images have matching segmentation overlays. Check overlay folder/name pattern.")

    scores = []
    for fam, g in pool_meta.groupby("family_id"):
        valid = g[g["folder_class"].isin(EYE_FOLDERS)]
        n_classes = len(set(valid["folder_class"]))
        min_per_class = int(valid.groupby("folder_class").size().min()) if n_classes else 0
        scores.append({"family_id": str(fam), "n_classes": n_classes, "min_per_class": min_per_class, "total": len(valid)})

    score_df = pd.DataFrame(scores)
    complete = score_df[(score_df["n_classes"] >= 4) & (score_df["min_per_class"] >= 1)].copy()
    fam_pool = complete if len(complete) >= families else score_df.copy()

    fams = fam_pool["family_id"].tolist()
    rng.shuffle(fams)
    chosen_fams = fams[:families]

    parts = []
    for fam in chosen_fams:
        fam_df = pool_meta[pool_meta["family_id"].astype(str) == str(fam)]
        for folder in sorted(EYE_FOLDERS):
            sub = fam_df[fam_df["folder_class"] == folder]
            if sub.empty:
                continue
            sub = sub.sample(frac=1, random_state=int(rng.integers(0, 1_000_000)))
            parts.append(sub.head(images_per_iris))

    if not parts:
        raise RuntimeError("No sample images selected.")

    selected = pd.concat(parts, ignore_index=True).drop_duplicates("stem").reset_index(drop=True)
    
    #if len(selected) >= 20:
    # selected = selected.sample(n=19, random_state=seed).reset_index(drop=True)
    return selected


# ---------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------

def artifact_files_for_kind(repo_root: Path, kind: str) -> list[Path]:
    if kind == "segmentation":
        return list_files(repo_root / "data/segmentation_daugman/overlays", IMAGE_EXT)
    if kind == "normalized":
        return list_files(repo_root / "data/normalized/daugman_strict_v3/images", IMAGE_EXT | ARRAY_EXT)
    if kind == "mask":
        return list_files(repo_root / "data/normalized/daugman_strict_v3/masks", IMAGE_EXT | ARRAY_EXT)
    if kind == "masked":
        return list_files(repo_root / "data/normalized/daugman_strict_v3/masked_previews", IMAGE_EXT | ARRAY_EXT)
    if kind == "gabor":
        return list_files(repo_root / "data/features/daugman_strict_v3/gabor_codes", IMAGE_EXT | ARRAY_EXT)
    return []


def build_artifact_sheet(selected: pd.DataFrame, files: list[Path], kind: str, title: str, out_sheet: Path, render_dir: Path) -> list[dict]:
    items: list[tuple[Path, str]] = []
    logs: list[dict] = []
    used: set[Path] = set()

    for _, row in selected.iterrows():
        stem = str(row["stem"])
        filename = str(row["filename"])
        found = exact_find_by_stem(files, stem, used=used)

        if found:
            used.add(found)
            if found.suffix.lower() in ARRAY_EXT:
                view = render_array_file(found, render_dir / f"{stem}_{kind}_{found.stem}.png")
            else:
                view = found
            label = f'{row["family_id"]}/{row["folder_class"]}/{filename}'
            items.append((view, label))
            logs.append({"kind": kind, "filename": filename, "stem": stem, "found": True, "source_path": str(found), "view_path": str(view)})
        else:
            logs.append({"kind": kind, "filename": filename, "stem": stem, "found": False, "source_path": "", "view_path": ""})

    save_contact_sheet(items, out_sheet, title)
    return logs


def build_raw_sheet(selected: pd.DataFrame, out_sheet: Path) -> None:
    items = []
    for _, row in selected.iterrows():
        p = Path(str(row["source_path"])) if row["source_path"] else None
        if p and p.exists():
            items.append((p, f'{row["family_id"]}/{row["folder_class"]}/{row["filename"]}'))
    save_contact_sheet(items, out_sheet, "01 - Selected raw images from strict subset")


# ---------------------------------------------------------------------
# Pairs and matching
# ---------------------------------------------------------------------

def relation_between(a: pd.Series, b: pd.Series) -> str:
    if str(a["iris_id"]) == str(b["iris_id"]):
        return "genuine"
    if str(a["subject_id"]) == str(b["subject_id"]):
        return "same_subject_different_eye"
    if str(a["family_id"]) == str(b["family_id"]):
        if str(a["eye"]) == str(b["eye"]):
            return "twin_same_eye"
        return "twin_cross_eye"
    return "unrelated"


def build_pairs(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a = selected.iloc[i]
            b = selected.iloc[j]
            rows.append({
                "a_filename": a["filename"],
                "b_filename": b["filename"],
                "a_stem": a["stem"],
                "b_stem": b["stem"],
                "relation": relation_between(a, b),
            })
    return pd.DataFrame(rows)


def filter_matching_results(repo_root: Path, selected: pd.DataFrame) -> pd.DataFrame:
    path = repo_root / "data/matching/daugman_strict_v3/matching_results.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    print(f"Reading matching results: {path}")
    df = pd.read_csv(path)
    relation_col = find_col(df, ["relation", "pair_relation", "comparison_relation", "pair_type", "type"])
    hd_col = find_col(df, ["hd", "hamming_distance", "masked_hd", "distance", "score", "best_hd"])
    if relation_col is None or hd_col is None:
        print("WARNING: cannot find relation/HD columns in matching results.")
        print(list(df.columns))
        return pd.DataFrame()

    selected_stems = set(selected["stem"].astype(str))
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    keep = []
    matched = []
    for _, row in df[text_cols].iterrows():
        text = " ".join(str(v) for v in row.values)
        found = sorted([s for s in selected_stems if s in text])
        keep.append(len(set(found)) >= 2)
        matched.append(",".join(found))

    sample = df.loc[keep].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["_matched_stems"] = [m for m, k in zip(matched, keep) if k]
    sample = sample.rename(columns={relation_col: "relation", hd_col: "hd"})
    sample["hd"] = pd.to_numeric(sample["hd"], errors="coerce")
    sample = sample.dropna(subset=["hd"])
    sample["matching_source"] = "filtered_from_final_matching_results"
    return sample


# ---------------------------------------------------------------------
# Live HD fallback
# ---------------------------------------------------------------------

def normalize_code(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[-1] <= 8 and arr.shape[0] > 8:
        arr = np.moveaxis(arr, -1, 0)
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported code shape: {arr.shape}")
    return arr > 0


def normalize_mask(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[-1] <= 8 and mask.shape[0] > 8:
            mask = np.moveaxis(mask, -1, 0)
        if mask.shape[0] == shape[0]:
            return mask > 0
        mask = np.any(mask > 0, axis=0)
    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")
    return np.broadcast_to((mask > 0)[None, :, :], shape)


def masked_hd(code_a, mask_a, code_b, mask_b, shifts=range(-16, 17, 2), min_common_bits=1000) -> tuple[float, int, int]:
    ca = normalize_code(code_a)
    cb0 = normalize_code(code_b)
    if ca.shape != cb0.shape:
        return float("nan"), 0, 0
    ma = normalize_mask(mask_a, ca.shape)
    mb0 = normalize_mask(mask_b, cb0.shape)
    best, best_shift, best_common = None, 0, 0
    for s in shifts:
        cb = np.roll(cb0, s, axis=2)
        mb = np.roll(mb0, s, axis=2)
        common = ma & mb
        n = int(np.count_nonzero(common))
        if n < min_common_bits:
            continue
        hd = float(np.count_nonzero((ca != cb) & common) / n)
        if best is None or hd < best:
            best, best_shift, best_common = hd, int(s), n
    if best is None:
        return float("nan"), 0, 0
    return best, best_shift, best_common


def compute_live_hd(repo_root: Path, selected: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    code_files = artifact_files_for_kind(repo_root, "gabor")
    mask_files = artifact_files_for_kind(repo_root, "mask")
    lookup = {}
    for _, row in selected.iterrows():
        stem = str(row["stem"])
        code = exact_find_by_stem(code_files, stem)
        mask = exact_find_by_stem(mask_files, stem)
        if code and mask:
            lookup[stem] = {"code": code, "mask": mask}
    if len(lookup) < 2:
        return pd.DataFrame()

    rows = []
    for _, p in pairs.iterrows():
        a, b = p["a_stem"], p["b_stem"]
        if a not in lookup or b not in lookup:
            continue
        try:
            hd, shift, common = masked_hd(
                load_array_file(lookup[a]["code"]),
                load_array_file(lookup[a]["mask"]),
                load_array_file(lookup[b]["code"]),
                load_array_file(lookup[b]["mask"]),
            )
        except Exception:
            hd, shift, common = float("nan"), 0, 0
        row = p.to_dict()
        row.update({"hd": hd, "best_shift": shift, "common_bits": common, "matching_source": "live_from_final_feature_codes"})
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.dropna(subset=["hd"])
    return out


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_pair_counts(pairs: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = pairs["relation"].value_counts().rename_axis("relation").reset_index(name="count")
    counts.to_csv(out_dir / "sample_pair_counts.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.bar(counts["relation"], counts["count"])
    plt.title("Demo sample: pair counts by relation")
    plt.xlabel("Relation")
    plt.ylabel("Number of pairs")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_pair_counts.png", dpi=200)
    plt.close()


def plot_matching(df: pd.DataFrame, out_dir: Path) -> dict:
    metrics = {}
    if df.empty:
        return metrics
    df = df.dropna(subset=["hd"]).copy()
    if df.empty:
        return metrics
    stats = df.groupby("relation")["hd"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index().sort_values("mean")
    stats.to_csv(out_dir / "sample_relation_stats.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(stats["relation"], stats["mean"])
    plt.title("Demo sample: mean Hamming Distance by relation")
    plt.xlabel("Relation")
    plt.ylabel("Mean Hamming Distance")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_mean_hd_by_relation.png", dpi=200)
    plt.close()

    order = ["genuine", "twin_same_eye", "twin_cross_eye", "same_subject_different_eye", "same_subj_diff_eye", "unrelated"]
    rels = list(df["relation"].dropna().unique())
    ordered = [r for r in order if r in rels] + [r for r in rels if r not in order]

    plt.figure(figsize=(10, 5))
    for rel in ordered:
        vals = df.loc[df["relation"] == rel, "hd"].dropna()
        if len(vals):
            plt.hist(vals, bins=min(20, max(5, len(vals))), alpha=0.55, density=True, label=f"{rel} (n={len(vals)})")
    plt.title("Demo sample: Hamming Distance distributions")
    plt.xlabel("Hamming Distance")
    plt.ylabel("Density")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "sample_hd_histograms.png", dpi=200)
    plt.close()

    box_data, box_labels = [], []
    for rel in ordered:
        vals = df.loc[df["relation"] == rel, "hd"].dropna().values
        if len(vals):
            box_data.append(vals)
            box_labels.append(rel)
    if box_data:
        plt.figure(figsize=(10, 5))
        plt.boxplot(box_data, labels=box_labels, showmeans=True)
        plt.title("Demo sample: HD boxplot by relation")
        plt.xlabel("Relation")
        plt.ylabel("Hamming Distance")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "sample_hd_boxplot.png", dpi=200)
        plt.close()

    plt.figure(figsize=(9, 5))
    plt.scatter(range(len(df)), df["hd"])
    plt.title("Demo sample: pair scores")
    plt.xlabel("Pair index")
    plt.ylabel("Hamming Distance")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_pair_scores_scatter.png", dpi=200)
    plt.close()

    df["label"] = (df["relation"] == "genuine").astype(int)
    pos, neg = int(df["label"].sum()), int((1 - df["label"]).sum())
    metrics["genuine_pairs"] = pos
    metrics["impostor_pairs"] = neg
    if pos > 0 and neg > 0:
        rows = []
        for t in np.sort(df["hd"].unique()):
            accept = df["hd"] <= t
            tp = int((accept & (df["label"] == 1)).sum())
            fp = int((accept & (df["label"] == 0)).sum())
            fn = int(((~accept) & (df["label"] == 1)).sum())
            far = fp / neg
            frr = fn / pos
            gar = tp / pos
            rows.append({"threshold": t, "FAR": far, "FRR": frr, "GAR": gar})
        curve = pd.DataFrame(rows).sort_values("FAR")
        curve.to_csv(out_dir / "sample_verification_curve.csv", index=False)
        idx = (curve["FAR"] - curve["FRR"]).abs().idxmin()
        metrics["EER"] = float((curve.loc[idx, "FAR"] + curve.loc[idx, "FRR"]) / 2)
        metrics["EER_threshold"] = float(curve.loc[idx, "threshold"])
        metrics["AUC"] = float(np.trapz(curve["GAR"], curve["FAR"])) if len(curve) >= 2 else None

        plt.figure(figsize=(6, 5))
        plt.plot(curve["FAR"], curve["GAR"], marker="o", markersize=3)
        plt.title("Demo sample: ROC curve")
        plt.xlabel("FAR")
        plt.ylabel("GAR")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "sample_roc_curve.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(curve["FAR"], curve["FRR"], marker="o", markersize=3)
        plt.title("Demo sample: DET-style curve")
        plt.xlabel("FAR")
        plt.ylabel("FRR")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "sample_det_curve.png", dpi=200)
        plt.close()

    with open(out_dir / "sample_verification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def write_report(out_dir: Path, selected: pd.DataFrame, artifact_log: pd.DataFrame, pairs: pd.DataFrame, matching: pd.DataFrame, metrics: dict) -> None:
    selected_txt = selected[["family_id", "folder_class", "filename", "iris_id", "subject_id"]].to_string(index=False)
    pair_counts_txt = pairs["relation"].value_counts().to_string()
    if matching.empty:
        stats_txt = "No sample-specific matching rows found."
        matching_source = "not_available"
    else:
        stats_txt = matching.groupby("relation")["hd"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index().sort_values("mean").to_string(index=False)
        matching_source = matching["matching_source"].iloc[0] if "matching_source" in matching.columns else "unknown"
    missing = artifact_log[artifact_log["found"] == False] if not artifact_log.empty else pd.DataFrame()
    missing_txt = missing[["kind", "filename", "stem"]].to_string(index=False) if not missing.empty else "No missing artifacts."

    text = f"""# Demo orale strict sample v3

Questa demo seleziona solo immagini della subset strict che hanno un overlay di segmentazione corrispondente.

## Sample selezionato

Numero immagini: **{len(selected)}**

```text
{selected_txt}
```

## Matching source

```text
{matching_source}
```

## Immagini da mostrare

```bash
open demo_orale_strict_sample_v3/reference_images/01_raw_selected.png
open demo_orale_strict_sample_v3/reference_images/02_segmentation_true_overlays.png
open demo_orale_strict_sample_v3/reference_images/03_normalized_strips.png
open demo_orale_strict_sample_v3/reference_images/04_masks.png
open demo_orale_strict_sample_v3/reference_images/05_masked_normalized_strips.png
open demo_orale_strict_sample_v3/reference_images/06_gabor_codes.png
```

## Pair counts del sample

```text
{pair_counts_txt}
```

## Sample-specific matching statistics

```text
{stats_txt}
```

## Verification metrics del sample

```json
{json.dumps(metrics, indent=2)}
```

## Plot del sample

```bash
open demo_orale_strict_sample_v3/plots/sample_pair_counts.png
open demo_orale_strict_sample_v3/plots/sample_mean_hd_by_relation.png
open demo_orale_strict_sample_v3/plots/sample_hd_histograms.png
open demo_orale_strict_sample_v3/plots/sample_hd_boxplot.png
open demo_orale_strict_sample_v3/plots/sample_pair_scores_scatter.png
open demo_orale_strict_sample_v3/plots/sample_roc_curve.png
open demo_orale_strict_sample_v3/plots/sample_det_curve.png
```

## Artifact mancanti

```text
{missing_txt}
```

## Frase pronta

> For this demo, we randomly select a small subset of images from the final Daugman-style strict subset, but only among samples for which the true final segmentation overlay is available. Then we retrieve the corresponding normalized strips, masks, masked strips and feature codes. Finally, we compute or retrieve Hamming Distance scores for the selected samples and generate sample-specific plots. These plots are only used to demonstrate the pipeline behavior; the official quantitative results are those reported in the paper.
"""
    (out_dir / "DEMO_REPORT.md").write_text(text, encoding="utf-8")


def open_outputs(out_dir: Path) -> None:
    for p in [
        out_dir / "DEMO_REPORT.md",
        out_dir / "reference_images/01_raw_selected.png",
        out_dir / "reference_images/02_segmentation_true_overlays.png",
        out_dir / "reference_images/03_normalized_strips.png",
        out_dir / "reference_images/04_masks.png",
        out_dir / "reference_images/05_masked_normalized_strips.png",
        out_dir / "reference_images/06_gabor_codes.png",
        out_dir / "plots/sample_hd_histograms.png",
    ]:
        if p.exists():
            subprocess.run(["open", str(p)])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    repo_root = args.repo_root
    out_dir = args.out_dir if args.out_dir else repo_root / "demo_orale_strict_sample_v3"
    ref_dir = out_dir / "reference_images"
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    render_dir = out_dir / "rendered_arrays"
    reset_dir(out_dir, clean=not args.no_clean)
    ref_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    strict_metadata = repo_root / "data/metadata/metadata_daugman_strict.csv"
    overlay_files = artifact_files_for_kind(repo_root, "segmentation")

    print("\n=== DEMO ORALE STRICT SAMPLE V3 ===")
    print(f"Repo root:        {repo_root}")
    print(f"Strict metadata:  {strict_metadata}")
    print(f"Overlay files:    {len(overlay_files)}")
    print(f"Output dir:       {out_dir}")

    meta = load_strict_metadata(repo_root, strict_metadata, overlay_files)
    print(f"Strict rows total:          {len(meta)}")
    print(f"Rows with true overlay:     {int(meta['has_segmentation_overlay'].sum())}")

    selected = select_demo_sample(meta, args.families, args.images_per_iris, args.seed, require_overlays=not args.allow_missing_overlays)
    selected.to_csv(tables_dir / "selected_strict_sample.csv", index=False)
    print(f"\nSelected images: {len(selected)}")
    print(selected[["family_id", "folder_class", "filename", "iris_id", "has_segmentation_overlay"]].to_string(index=False))

    print("\n[1] Building raw image sheet...")
    build_raw_sheet(selected, ref_dir / "01_raw_selected.png")

    all_logs = []
    steps = [
        ("segmentation", "02 - True Daugman-style strict segmentation overlays", ref_dir / "02_segmentation_true_overlays.png"),
        ("normalized", "03 - Normalized iris strips", ref_dir / "03_normalized_strips.png"),
        ("mask", "04 - Final conservative masks", ref_dir / "04_masks.png"),
        ("masked", "05 - Masked normalized iris strips", ref_dir / "05_masked_normalized_strips.png"),
        ("gabor", "06 - Gabor-style feature/code previews", ref_dir / "06_gabor_codes.png"),
    ]
    for kind, title, out_sheet in steps:
        files = artifact_files_for_kind(repo_root, kind)
        print(f"\n[{kind}] candidate files: {len(files)}")
        logs = build_artifact_sheet(selected, files, kind, title, out_sheet, render_dir / kind)
        all_logs.extend(logs)
        found = sum(1 for x in logs if x["found"])
        print(f"Found {kind}: {found}/{len(selected)}")

    artifact_log = pd.DataFrame(all_logs)
    artifact_log.to_csv(tables_dir / "artifact_lookup_log.csv", index=False)

    bad_seg = artifact_log[(artifact_log["kind"] == "segmentation") & (artifact_log["found"] == True)].copy()
    for _, r in bad_seg.iterrows():
        if str(r["stem"]).lower() not in str(r["source_path"]).lower():
            raise RuntimeError(f"BUG PREVENTED: segmentation for {r['stem']} matched wrong path {r['source_path']}")

    print("\n[Pairs] Building sample pairs...")
    pairs = build_pairs(selected)
    pairs.to_csv(tables_dir / "sample_pairs.csv", index=False)
    print(pairs["relation"].value_counts().to_string())

    print("\n[Matching] Filtering final matching results...")
    matching = filter_matching_results(repo_root, selected)
    if matching.empty:
        print("No rows found by filtering. Trying live HD from final feature codes...")
        matching = compute_live_hd(repo_root, selected, pairs)
    if matching.empty:
        print("WARNING: no sample-specific HD scores produced.")
    else:
        print(f"Sample matching rows: {len(matching)}")
        if "matching_source" in matching.columns:
            print(f"Matching source: {matching['matching_source'].iloc[0]}")
    matching.to_csv(tables_dir / "sample_matching_results.csv", index=False)

    print("\n[Plots] Generating plots...")
    plot_pair_counts(pairs, plots_dir)
    metrics = plot_matching(matching, plots_dir)

    summary = {
        "seed": args.seed,
        "families": args.families,
        "images_per_iris": args.images_per_iris,
        "selected_images": int(len(selected)),
        "selected_families": sorted(map(str, selected["family_id"].unique())),
        "strict_rows_total": int(len(meta)),
        "rows_with_true_overlay": int(meta["has_segmentation_overlay"].sum()),
        "sample_pairs": int(len(pairs)),
        "sample_matching_rows": int(len(matching)),
        "metrics": metrics,
    }
    with open(out_dir / "demo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_report(out_dir, selected, artifact_log, pairs, matching, metrics)

    print("\n=== DONE ===")
    print(f"Report: {out_dir / 'DEMO_REPORT.md'}")
    print("\nOpen:")
    print(f'open "{out_dir / "DEMO_REPORT.md"}"')
    print(f'open "{ref_dir / "02_segmentation_true_overlays.png"}"')
    print(f'open "{tables_dir / "artifact_lookup_log.csv"}"')

    if args.open:
        open_outputs(out_dir)


if __name__ == "__main__":
    main()
