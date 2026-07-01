from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "daugman_strict_v3" / "features_report.csv"
DEFAULT_INPUT_MATCHING = PROJECT_ROOT / "data" / "matching" / "daugman_strict_v3" / "matching_results.csv.gz"
DEFAULT_INPUT_CLOSED = PROJECT_ROOT / "data" / "identification_closed_set" / "daugman_strict_v3" / "closed_set_results.csv"
DEFAULT_INPUT_TOPK = PROJECT_ROOT / "data" / "identification_closed_set" / "daugman_strict_v3" / "closed_set_topk.csv.gz"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "failure_analysis" / "daugman_strict_v3"

TOP_N = 40

THUMB_W = 220
THUMB_H = 70
LABEL_H = 58
COLS = 2


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


def as_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_number(x):
    try:
        x = float(x)
        if math.isfinite(x):
            return x
        return None
    except Exception:
        return None


def fmt_number(x, digits=4):
    value = safe_number(x)
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"


def stats(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()

    if len(arr) == 0:
        return None

    return {
        "n": int(len(arr)),
        "min": safe_number(np.min(arr)),
        "q1": safe_number(np.quantile(arr, 0.25)),
        "median": safe_number(np.median(arr)),
        "mean": safe_number(np.mean(arr)),
        "q3": safe_number(np.quantile(arr, 0.75)),
        "max": safe_number(np.max(arr)),
        "std": safe_number(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def corr(x, y):
    tmp = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(tmp) < 3:
        return None

    return safe_number(tmp["x"].corr(tmp["y"]))


def read_features(input_features: Path):
    dtype_map = {
        "family_id": str,
        "twin_id": str,
        "eye": str,
        "subject_id": str,
        "iris_id": str,
        "relative_path": str,
        "normalized_path": str,
        "mask_path": str,
        "masked_preview_path": str,
        "code_path": str,
        "code_preview_path": str,
    }

    df = pd.read_csv(input_features, dtype=dtype_map)

    df = df[as_bool(df["feature_ok"])].copy()
    df["valid_bit_ratio"] = pd.to_numeric(df["valid_bit_ratio"], errors="coerce")

    feature_by_rel = {
        row["relative_path"]: row.to_dict()
        for _, row in df.iterrows()
    }

    gallery_by_iris = {}

    for iris_id, group in df.groupby("iris_id"):
        group = group.copy()
        group = group.sort_values(
            ["valid_bit_ratio", "relative_path"],
            ascending=[False, True],
        )
        gallery_by_iris[str(iris_id)] = group.iloc[0].to_dict()

    return df, feature_by_rel, gallery_by_iris


def valid_path(value):
    if value is None:
        return None

    value = str(value)

    if value == "" or value.lower() == "nan":
        return None

    path = resolve_from_project(Path(value))

    if path.exists():
        return path

    return None


def visual_path(relative_path, feature_by_rel):
    """
    Preferiamo la masked preview, perché mostra la parte realmente usata.
    Fallback: normalized image, poi code preview.
    """
    row = feature_by_rel.get(str(relative_path))

    if row is None:
        return None

    preferred_cols = [
        "masked_preview_path",
        "normalized_path",
        "code_preview_path",
    ]

    for col in preferred_cols:
        path = valid_path(row.get(col))
        if path is not None:
            return path

    return None


def load_thumb(relative_path, feature_by_rel, size=(THUMB_W, THUMB_H)):
    path = visual_path(relative_path, feature_by_rel)

    if path is None:
        img = Image.new("RGB", size, color=(230, 230, 230))
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), "image not found", fill=(0, 0, 0))
        return img

    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size)
        return img
    except Exception:
        img = Image.new("RGB", size, color=(230, 230, 230))
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), "load error", fill=(0, 0, 0))
        return img


def draw_text(draw, xy, text, font, max_len=56):
    text = str(text)
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def make_pair_contact_sheet(df, output_path, title, feature_by_rel, n=TOP_N):
    if len(df) == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.head(n).copy()

    font = ImageFont.load_default()

    card_w = THUMB_W * 2
    card_h = THUMB_H + LABEL_H

    rows = math.ceil(len(df) / COLS)

    sheet = Image.new(
        "RGB",
        (COLS * card_w, rows * card_h + 34),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)

    y0 = 34

    for idx, (_, row) in enumerate(df.iterrows()):
        x = (idx % COLS) * card_w
        y = y0 + (idx // COLS) * card_h

        img1 = load_thumb(row["image_1"], feature_by_rel)
        img2 = load_thumb(row["image_2"], feature_by_rel)

        sheet.paste(img1, (x, y))
        sheet.paste(img2, (x + THUMB_W, y))

        d = ImageDraw.Draw(sheet)

        label1 = f"{row['relation']} | HD={fmt_number(row['hamming_distance'])}"
        label2 = f"1: {row['image_1']}"
        label3 = f"2: {row['image_2']}"

        draw_text(d, (x + 4, y + THUMB_H + 4), label1, font)
        draw_text(d, (x + 4, y + THUMB_H + 20), label2, font)
        draw_text(d, (x + 4, y + THUMB_H + 36), label3, font)

    sheet.save(output_path)
    print(f"Salvata contact sheet: {project_display_path(output_path)}")


def make_identification_contact_sheet(rows, output_path, title, feature_by_rel, n=TOP_N):
    if len(rows) == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows[:n]

    font = ImageFont.load_default()

    item_w = THUMB_W * 3
    item_h = THUMB_H + LABEL_H

    rows_count = len(rows)

    sheet = Image.new(
        "RGB",
        (item_w, rows_count * item_h + 34),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)

    y0 = 34

    for idx, item in enumerate(rows):
        x = 0
        y = y0 + idx * item_h

        probe_img = load_thumb(item["probe"], feature_by_rel)
        true_img = load_thumb(item["true_gallery"], feature_by_rel)
        other_img = load_thumb(item["other_gallery"], feature_by_rel)

        sheet.paste(probe_img, (x, y))
        sheet.paste(true_img, (x + THUMB_W, y))
        sheet.paste(other_img, (x + 2 * THUMB_W, y))

        d = ImageDraw.Draw(sheet)

        draw_text(d, (x + 4, y + THUMB_H + 4), f"Probe: {item['probe']}", font)
        draw_text(d, (x + THUMB_W + 4, y + THUMB_H + 4), f"True: {item['true_gallery']}", font)
        draw_text(d, (x + 2 * THUMB_W + 4, y + THUMB_H + 4), f"Other: {item['other_gallery']}", font)

        draw_text(d, (x + 4, y + THUMB_H + 22), item["line1"], font, max_len=90)
        draw_text(d, (x + 4, y + THUMB_H + 40), item["line2"], font, max_len=90)

    sheet.save(output_path)
    print(f"Salvata contact sheet: {project_display_path(output_path)}")


def add_pair_feature_columns(df, feature_by_rel):
    def get_valid(rel):
        row = feature_by_rel.get(str(rel))
        if row is None:
            return np.nan
        return safe_number(row.get("valid_bit_ratio"))

    df = df.copy()
    df["valid_bit_ratio_1"] = df["image_1"].map(get_valid)
    df["valid_bit_ratio_2"] = df["image_2"].map(get_valid)
    df["valid_bit_ratio_mean"] = df[["valid_bit_ratio_1", "valid_bit_ratio_2"]].mean(axis=1)
    df["valid_bit_ratio_min"] = df[["valid_bit_ratio_1", "valid_bit_ratio_2"]].min(axis=1)

    return df


def read_matching(input_matching: Path, feature_by_rel):
    df = pd.read_csv(input_matching)
    df = df[as_bool(df["match_ok"])].copy()
    df["hamming_distance"] = pd.to_numeric(df["hamming_distance"], errors="coerce")
    df = add_pair_feature_columns(df, feature_by_rel)
    return df


def read_closed(input_closed: Path):
    dtype_map = {
        "probe_iris_id": str,
        "probe_relative_path": str,
    }

    df = pd.read_csv(input_closed, dtype=dtype_map)

    df["rank1_correct"] = as_bool(df["rank1_correct"])
    df["true_rank"] = pd.to_numeric(df["true_rank"], errors="coerce")
    df["margin_best_impostor_minus_true"] = pd.to_numeric(
        df["margin_best_impostor_minus_true"],
        errors="coerce",
    )
    df["margin_best_twin_minus_true"] = pd.to_numeric(
        df["margin_best_twin_minus_true"],
        errors="coerce",
    )
    df["min_twin_rank"] = pd.to_numeric(df["min_twin_rank"], errors="coerce")

    return df


def read_topk(input_topk: Path):
    dtype_map = {
        "probe_relative_path": str,
        "gallery_relative_path": str,
        "probe_iris_id": str,
        "gallery_iris_id": str,
    }

    df = pd.read_csv(input_topk, dtype=dtype_map)

    for col in ["is_true", "is_twin", "is_twin_same_eye", "is_twin_cross_eye"]:
        df[col] = as_bool(df[col])

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    return df


def count_dict(series):
    counts = series.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def run_failure_analysis(
    input_features: Path,
    input_matching: Path,
    input_closed: Path,
    input_topk: Path,
    output_dir: Path,
    top_n: int,
):
    for path, label in [
        (input_features, "features"),
        (input_matching, "matching"),
        (input_closed, "closed-set results"),
        (input_topk, "closed-set top-k"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Input {label} non trovato: {path}")

    contact_dir = output_dir / "contact_sheets"
    summary_json = output_dir / "failure_analysis_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    print("=== FAILURE ANALYSIS ===")
    print(f"Features:     {project_display_path(input_features)}")
    print(f"Matching:     {project_display_path(input_matching)}")
    print(f"Closed-set:   {project_display_path(input_closed)}")
    print(f"Top-k:        {project_display_path(input_topk)}")
    print(f"Output dir:   {project_display_path(output_dir)}")
    print()

    features_df, feature_by_rel, gallery_by_iris = read_features(input_features)
    matching = read_matching(input_matching, feature_by_rel)
    closed = read_closed(input_closed)
    topk = read_topk(input_topk)

    print(f"Feature templates: {len(features_df)}")
    print(f"Matching rows:     {len(matching)}")
    print(f"Closed-set probes: {len(closed)}")
    print()

    hard_genuine = (
        matching[matching["relation"] == "genuine"]
        .sort_values("hamming_distance", ascending=False)
        .head(top_n)
        .copy()
    )

    low_unrelated = (
        matching[matching["relation"] == "unrelated"]
        .sort_values("hamming_distance", ascending=True)
        .head(top_n)
        .copy()
    )

    low_twin_same_eye = (
        matching[matching["relation"] == "twin_same_eye"]
        .sort_values("hamming_distance", ascending=True)
        .head(top_n)
        .copy()
    )

    low_twin_cross_eye = (
        matching[matching["relation"] == "twin_cross_eye"]
        .sort_values("hamming_distance", ascending=True)
        .head(top_n)
        .copy()
    )

    rank1_errors = (
        closed[closed["rank1_correct"] == False]
        .sort_values("margin_best_impostor_minus_true", ascending=True)
        .copy()
    )

    twin_near_misses = (
        closed[
            (closed["min_twin_rank"] <= 5)
            | (closed["margin_best_twin_minus_true"] < 0.03)
        ]
        .sort_values(["min_twin_rank", "margin_best_twin_minus_true"], ascending=[True, True])
        .copy()
    )

    hard_genuine.to_csv(output_dir / "hard_genuine_pairs.csv", index=False)
    low_unrelated.to_csv(output_dir / "low_unrelated_pairs.csv", index=False)
    low_twin_same_eye.to_csv(output_dir / "low_twin_same_eye_pairs.csv", index=False)
    low_twin_cross_eye.to_csv(output_dir / "low_twin_cross_eye_pairs.csv", index=False)
    rank1_errors.to_csv(output_dir / "rank1_errors.csv", index=False)
    twin_near_misses.to_csv(output_dir / "twin_near_misses.csv", index=False)

    make_pair_contact_sheet(
        hard_genuine,
        contact_dir / "hard_genuine_pairs.png",
        "Hard genuine pairs: highest HD",
        feature_by_rel,
        n=top_n,
    )

    make_pair_contact_sheet(
        low_unrelated,
        contact_dir / "low_unrelated_pairs.png",
        "Suspicious unrelated pairs: lowest HD",
        feature_by_rel,
        n=top_n,
    )

    make_pair_contact_sheet(
        low_twin_same_eye,
        contact_dir / "low_twin_same_eye_pairs.png",
        "Twin same-eye pairs: lowest HD",
        feature_by_rel,
        n=top_n,
    )

    make_pair_contact_sheet(
        low_twin_cross_eye,
        contact_dir / "low_twin_cross_eye_pairs.png",
        "Twin cross-eye pairs: lowest HD",
        feature_by_rel,
        n=top_n,
    )

    rank_error_items = []

    for _, row in rank1_errors.head(top_n).iterrows():
        probe_rel = row["probe_relative_path"]
        true_gallery = gallery_by_iris.get(str(row["probe_iris_id"]), {})
        true_rel = true_gallery.get("relative_path", "")

        top1 = topk[
            (topk["probe_relative_path"] == probe_rel)
            & (topk["rank"] == 1)
        ]

        if len(top1) > 0:
            top1_row = top1.iloc[0]
            other_rel = top1_row["gallery_relative_path"]
            relation = top1_row["relation_to_probe"]
            score = top1_row["score"]
        else:
            other_rel = ""
            relation = row.get("rank1_relation_to_probe", "")
            score = row.get("rank1_score", np.nan)

        rank_error_items.append(
            {
                "probe": probe_rel,
                "true_gallery": true_rel,
                "other_gallery": other_rel,
                "line1": f"true_rank={fmt_number(row['true_rank'], 0)} margin={fmt_number(row['margin_best_impostor_minus_true'])}",
                "line2": f"rank1_relation={relation} rank1_score={fmt_number(score)}",
            }
        )

    make_identification_contact_sheet(
        rank_error_items,
        contact_dir / "rank1_errors.png",
        "Closed-set Rank-1 errors: probe / true gallery / wrong rank-1",
        feature_by_rel,
        n=top_n,
    )

    twin_items = []

    for _, row in twin_near_misses.head(top_n).iterrows():
        probe_rel = row["probe_relative_path"]
        true_gallery = gallery_by_iris.get(str(row["probe_iris_id"]), {})
        true_rel = true_gallery.get("relative_path", "")

        probe_topk = topk[topk["probe_relative_path"] == probe_rel].copy()
        probe_twins = probe_topk[probe_topk["is_twin"] == True].sort_values("rank")

        if len(probe_twins) == 0:
            continue

        twin_row = probe_twins.iloc[0]
        twin_rel = twin_row["gallery_relative_path"]

        twin_items.append(
            {
                "probe": probe_rel,
                "true_gallery": true_rel,
                "other_gallery": twin_rel,
                "line1": f"twin_rank={fmt_number(twin_row['rank'], 0)} twin_score={fmt_number(twin_row['score'])}",
                "line2": f"true_rank={fmt_number(row['true_rank'], 0)} margin_twin_true={fmt_number(row['margin_best_twin_minus_true'])}",
            }
        )

    make_identification_contact_sheet(
        twin_items,
        contact_dir / "twin_near_misses.png",
        "Twin near misses: probe / true gallery / closest twin",
        feature_by_rel,
        n=top_n,
    )

    genuine_all = matching[matching["relation"] == "genuine"].copy()
    unrelated_all = matching[matching["relation"] == "unrelated"].copy()

    rank1_error_relation_counts = (
        count_dict(rank1_errors["best_impostor_relation_to_probe"])
        if "best_impostor_relation_to_probe" in rank1_errors.columns
        else {}
    )

    closed["probe_valid_bit_ratio"] = closed["probe_relative_path"].map(
        lambda rel: feature_by_rel.get(str(rel), {}).get("valid_bit_ratio", np.nan)
    )
    closed["probe_valid_bit_ratio"] = pd.to_numeric(closed["probe_valid_bit_ratio"], errors="coerce")

    rank1_errors["probe_valid_bit_ratio"] = rank1_errors["probe_relative_path"].map(
        lambda rel: feature_by_rel.get(str(rel), {}).get("valid_bit_ratio", np.nan)
    )
    rank1_errors["probe_valid_bit_ratio"] = pd.to_numeric(rank1_errors["probe_valid_bit_ratio"], errors="coerce")

    summary = {
        "inputs": {
            "features": project_display_path(input_features),
            "matching": project_display_path(input_matching),
            "closed_set_results": project_display_path(input_closed),
            "closed_set_topk": project_display_path(input_topk),
        },
        "counts": {
            "feature_templates": int(len(features_df)),
            "matching_rows": int(len(matching)),
            "closed_set_probes": int(len(closed)),
            "rank1_errors": int(len(rank1_errors)),
            "twin_near_misses": int(len(twin_near_misses)),
        },
        "pairwise_extreme_cases": {
            "hard_genuine_top_n": int(len(hard_genuine)),
            "hard_genuine_hd_stats": stats(hard_genuine["hamming_distance"]),
            "low_unrelated_top_n": int(len(low_unrelated)),
            "low_unrelated_hd_stats": stats(low_unrelated["hamming_distance"]),
            "low_twin_same_eye_top_n": int(len(low_twin_same_eye)),
            "low_twin_same_eye_hd_stats": stats(low_twin_same_eye["hamming_distance"]),
            "low_twin_cross_eye_top_n": int(len(low_twin_cross_eye)),
            "low_twin_cross_eye_hd_stats": stats(low_twin_cross_eye["hamming_distance"]),
        },
        "rank1_error_analysis": {
            "rank1_errors": int(len(rank1_errors)),
            "rank1_error_relation_counts": rank1_error_relation_counts,
            "rank1_error_probe_valid_bit_ratio_stats": stats(rank1_errors["probe_valid_bit_ratio"]),
            "all_probe_valid_bit_ratio_stats": stats(closed["probe_valid_bit_ratio"]),
            "rank1_error_margin_stats": stats(rank1_errors["margin_best_impostor_minus_true"]),
        },
        "twin_near_miss_analysis": {
            "twin_near_misses": int(len(twin_near_misses)),
            "min_twin_rank_stats": stats(twin_near_misses["min_twin_rank"]),
            "margin_best_twin_minus_true_stats": stats(twin_near_misses["margin_best_twin_minus_true"]),
        },
        "correlations": {
            "genuine_hd_vs_valid_bit_ratio_mean": corr(
                genuine_all["hamming_distance"],
                genuine_all["valid_bit_ratio_mean"],
            ),
            "unrelated_hd_vs_valid_bit_ratio_mean": corr(
                unrelated_all["hamming_distance"],
                unrelated_all["valid_bit_ratio_mean"],
            ),
            "closed_set_margin_vs_probe_valid_bit_ratio": corr(
                closed["margin_best_impostor_minus_true"],
                closed["probe_valid_bit_ratio"],
            ),
            "closed_set_true_rank_vs_probe_valid_bit_ratio": corr(
                closed["true_rank"],
                closed["probe_valid_bit_ratio"],
            ),
        },
        "outputs": {
            "hard_genuine_pairs_csv": project_display_path(output_dir / "hard_genuine_pairs.csv"),
            "low_unrelated_pairs_csv": project_display_path(output_dir / "low_unrelated_pairs.csv"),
            "low_twin_same_eye_pairs_csv": project_display_path(output_dir / "low_twin_same_eye_pairs.csv"),
            "low_twin_cross_eye_pairs_csv": project_display_path(output_dir / "low_twin_cross_eye_pairs.csv"),
            "rank1_errors_csv": project_display_path(output_dir / "rank1_errors.csv"),
            "twin_near_misses_csv": project_display_path(output_dir / "twin_near_misses.csv"),
            "contact_sheets": {
                "hard_genuine_pairs": project_display_path(contact_dir / "hard_genuine_pairs.png"),
                "low_unrelated_pairs": project_display_path(contact_dir / "low_unrelated_pairs.png"),
                "low_twin_same_eye_pairs": project_display_path(contact_dir / "low_twin_same_eye_pairs.png"),
                "low_twin_cross_eye_pairs": project_display_path(contact_dir / "low_twin_cross_eye_pairs.png"),
                "rank1_errors": project_display_path(contact_dir / "rank1_errors.png"),
                "twin_near_misses": project_display_path(contact_dir / "twin_near_misses.png"),
            },
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print()
    print("=== FAILURE ANALYSIS SUMMARY ===")
    print(f"Hard genuine pairs:       {len(hard_genuine)}")
    print(f"Low unrelated pairs:      {len(low_unrelated)}")
    print(f"Low twin same-eye pairs:  {len(low_twin_same_eye)}")
    print(f"Low twin cross-eye pairs: {len(low_twin_cross_eye)}")
    print(f"Rank-1 errors:            {len(rank1_errors)}")
    print(f"Twin near misses:         {len(twin_near_misses)}")
    print()

    print("Rank-1 error relation counts:")
    for key, value in rank1_error_relation_counts.items():
        print(f"  {key}: {value}")

    print()
    print("Correlations:")
    for key, value in summary["correlations"].items():
        print(f"  {key}: {value}")

    print()
    print(f"Summary:        {project_display_path(summary_json)}")
    print(f"Contact sheets: {project_display_path(contact_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-features",
        type=Path,
        default=DEFAULT_INPUT_FEATURES,
        help="Path a features_report.csv.",
    )
    parser.add_argument(
        "--input-matching",
        type=Path,
        default=DEFAULT_INPUT_MATCHING,
        help="Path a matching_results.csv.gz.",
    )
    parser.add_argument(
        "--input-closed",
        type=Path,
        default=DEFAULT_INPUT_CLOSED,
        help="Path a closed_set_results.csv.",
    )
    parser.add_argument(
        "--input-topk",
        type=Path,
        default=DEFAULT_INPUT_TOPK,
        help="Path a closed_set_topk.csv.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Cartella output failure analysis.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="Numero massimo di esempi da salvare per categoria.",
    )

    args = parser.parse_args()

    run_failure_analysis(
        input_features=resolve_from_project(args.input_features),
        input_matching=resolve_from_project(args.input_matching),
        input_closed=resolve_from_project(args.input_closed),
        input_topk=resolve_from_project(args.input_topk),
        output_dir=resolve_from_project(args.output_dir),
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
