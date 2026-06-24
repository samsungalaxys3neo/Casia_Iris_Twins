from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FEATURES = PROJECT_ROOT / "data/features/daugman_strict_v3/features_report.csv"
INPUT_MATCHING = PROJECT_ROOT / "data/matching/daugman_strict_v3/matching_results.csv.gz"
INPUT_CLOSED = PROJECT_ROOT / "data/identification_closed_set/daugman_strict_v3/closed_set_results.csv"
INPUT_TOPK = PROJECT_ROOT / "data/identification_closed_set/daugman_strict_v3/closed_set_topk.csv.gz"

OUTPUT_DIR = PROJECT_ROOT / "data/failure_analysis/daugman_strict_v3"
CONTACT_DIR = OUTPUT_DIR / "contact_sheets"

SUMMARY_JSON = OUTPUT_DIR / "failure_analysis_summary.json"

TOP_N = 40

THUMB_W = 220
THUMB_H = 70
LABEL_H = 58
COLS = 2


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)


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
    tmp = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    }).dropna()

    if len(tmp) < 3:
        return None

    return safe_number(tmp["x"].corr(tmp["y"]))


def load_features():
    df = pd.read_csv(
        INPUT_FEATURES,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "subject_id": str,
            "iris_id": str,
            "relative_path": str,
        }
    )

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
            ascending=[False, True]
        )
        gallery_by_iris[str(iris_id)] = group.iloc[0].to_dict()

    return df, feature_by_rel, gallery_by_iris


def valid_path(value):
    if value is None:
        return None

    value = str(value)

    if value == "" or value.lower() == "nan":
        return None

    p = Path(value)

    if p.exists():
        return p

    return None


def visual_path(relative_path, feature_by_rel):
    """
    Preferiamo la masked preview, perché mostra la parte realmente usata.
    Fallback: normalized image, code preview, raw image.
    """
    row = feature_by_rel.get(str(relative_path))

    if row is None:
        return None

    preferred_cols = [
        "masked_preview_path",
        "normalized_path",
        "code_preview_path",
        "absolute_path",
    ]

    for col in preferred_cols:
        p = valid_path(row.get(col))
        if p is not None:
            return p

    return None


def load_thumb(relative_path, feature_by_rel, size=(THUMB_W, THUMB_H)):
    p = visual_path(relative_path, feature_by_rel)

    if p is None:
        img = Image.new("RGB", size, color=(230, 230, 230))
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), "image not found", fill=(0, 0, 0))
        return img

    try:
        img = Image.open(p).convert("RGB")
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

    df = df.head(n).copy()

    font = ImageFont.load_default()

    card_w = THUMB_W * 2
    card_h = THUMB_H + LABEL_H

    rows = math.ceil(len(df) / COLS)

    sheet = Image.new(
        "RGB",
        (COLS * card_w, rows * card_h + 34),
        color=(255, 255, 255)
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

        label1 = f"{row['relation']} | HD={float(row['hamming_distance']):.4f}"
        label2 = f"1: {row['image_1']}"
        label3 = f"2: {row['image_2']}"

        draw_text(d, (x + 4, y + THUMB_H + 4), label1, font)
        draw_text(d, (x + 4, y + THUMB_H + 20), label2, font)
        draw_text(d, (x + 4, y + THUMB_H + 36), label3, font)

    sheet.save(output_path)
    print(f"Salvata contact sheet: {output_path}")


def make_identification_contact_sheet(rows, output_path, title, feature_by_rel, n=TOP_N):
    if len(rows) == 0:
        return

    rows = rows[:n]

    font = ImageFont.load_default()

    item_w = THUMB_W * 3
    item_h = THUMB_H + LABEL_H

    rows_count = math.ceil(len(rows) / 1)

    sheet = Image.new(
        "RGB",
        (item_w, rows_count * item_h + 34),
        color=(255, 255, 255)
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
    print(f"Salvata contact sheet: {output_path}")


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


def main():
    ensure_dirs()

    print("=== FAILURE ANALYSIS ===")

    features_df, feature_by_rel, gallery_by_iris = load_features()

    matching = pd.read_csv(INPUT_MATCHING)
    matching = matching[as_bool(matching["match_ok"])].copy()
    matching["hamming_distance"] = pd.to_numeric(matching["hamming_distance"], errors="coerce")
    matching = add_pair_feature_columns(matching, feature_by_rel)

    closed = pd.read_csv(
        INPUT_CLOSED,
        dtype={
            "probe_iris_id": str,
            "probe_relative_path": str,
        }
    )

    closed["rank1_correct"] = as_bool(closed["rank1_correct"])
    closed["true_rank"] = pd.to_numeric(closed["true_rank"], errors="coerce")
    closed["margin_best_impostor_minus_true"] = pd.to_numeric(
        closed["margin_best_impostor_minus_true"],
        errors="coerce"
    )
    closed["margin_best_twin_minus_true"] = pd.to_numeric(
        closed["margin_best_twin_minus_true"],
        errors="coerce"
    )
    closed["min_twin_rank"] = pd.to_numeric(closed["min_twin_rank"], errors="coerce")

    topk = pd.read_csv(
        INPUT_TOPK,
        dtype={
            "probe_relative_path": str,
            "gallery_relative_path": str,
            "probe_iris_id": str,
            "gallery_iris_id": str,
        }
    )

    for col in ["is_true", "is_twin", "is_twin_same_eye", "is_twin_cross_eye"]:
        topk[col] = as_bool(topk[col])

    print(f"Feature templates: {len(features_df)}")
    print(f"Matching rows:     {len(matching)}")
    print(f"Closed-set probes: {len(closed)}")
    print()

    # Pairwise failure/suspicious cases
    hard_genuine = (
        matching[matching["relation"] == "genuine"]
        .sort_values("hamming_distance", ascending=False)
        .head(TOP_N)
        .copy()
    )

    low_unrelated = (
        matching[matching["relation"] == "unrelated"]
        .sort_values("hamming_distance", ascending=True)
        .head(TOP_N)
        .copy()
    )

    low_twin_same_eye = (
        matching[matching["relation"] == "twin_same_eye"]
        .sort_values("hamming_distance", ascending=True)
        .head(TOP_N)
        .copy()
    )

    low_twin_cross_eye = (
        matching[matching["relation"] == "twin_cross_eye"]
        .sort_values("hamming_distance", ascending=True)
        .head(TOP_N)
        .copy()
    )

    # Closed-set failure cases
    rank1_errors = (
        closed[closed["rank1_correct"] == False]
        .sort_values("margin_best_impostor_minus_true", ascending=True)
        .copy()
    )

    # Twin near-miss: gemello molto vicino o in top-5
    twin_near_misses = (
        closed[
            (closed["min_twin_rank"] <= 5) |
            (closed["margin_best_twin_minus_true"] < 0.03)
        ]
        .sort_values(["min_twin_rank", "margin_best_twin_minus_true"], ascending=[True, True])
        .copy()
    )

    # Salva CSV
    hard_genuine.to_csv(OUTPUT_DIR / "hard_genuine_pairs.csv", index=False)
    low_unrelated.to_csv(OUTPUT_DIR / "low_unrelated_pairs.csv", index=False)
    low_twin_same_eye.to_csv(OUTPUT_DIR / "low_twin_same_eye_pairs.csv", index=False)
    low_twin_cross_eye.to_csv(OUTPUT_DIR / "low_twin_cross_eye_pairs.csv", index=False)
    rank1_errors.to_csv(OUTPUT_DIR / "rank1_errors.csv", index=False)
    twin_near_misses.to_csv(OUTPUT_DIR / "twin_near_misses.csv", index=False)

    # Contact sheets pairwise
    make_pair_contact_sheet(
        hard_genuine,
        CONTACT_DIR / "hard_genuine_pairs.png",
        "Hard genuine pairs: highest HD",
        feature_by_rel,
        n=TOP_N,
    )

    make_pair_contact_sheet(
        low_unrelated,
        CONTACT_DIR / "low_unrelated_pairs.png",
        "Suspicious unrelated pairs: lowest HD",
        feature_by_rel,
        n=TOP_N,
    )

    make_pair_contact_sheet(
        low_twin_same_eye,
        CONTACT_DIR / "low_twin_same_eye_pairs.png",
        "Twin same-eye pairs: lowest HD",
        feature_by_rel,
        n=TOP_N,
    )

    make_pair_contact_sheet(
        low_twin_cross_eye,
        CONTACT_DIR / "low_twin_cross_eye_pairs.png",
        "Twin cross-eye pairs: lowest HD",
        feature_by_rel,
        n=TOP_N,
    )

    # Contact sheet rank1 errors
    rank_error_items = []

    for _, row in rank1_errors.head(TOP_N).iterrows():
        probe_rel = row["probe_relative_path"]
        true_gallery = gallery_by_iris.get(str(row["probe_iris_id"]), {})
        true_rel = true_gallery.get("relative_path", "")

        top1 = topk[
            (topk["probe_relative_path"] == probe_rel) &
            (topk["rank"] == 1)
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

        rank_error_items.append({
            "probe": probe_rel,
            "true_gallery": true_rel,
            "other_gallery": other_rel,
            "line1": f"true_rank={row['true_rank']} margin={row['margin_best_impostor_minus_true']:.4f}",
            "line2": f"rank1_relation={relation} rank1_score={float(score):.4f}",
        })

    make_identification_contact_sheet(
        rank_error_items,
        CONTACT_DIR / "rank1_errors.png",
        "Closed-set Rank-1 errors: probe / true gallery / wrong rank-1",
        feature_by_rel,
        n=TOP_N,
    )

    # Contact sheet twin near misses
    twin_items = []

    for _, row in twin_near_misses.head(TOP_N).iterrows():
        probe_rel = row["probe_relative_path"]
        true_gallery = gallery_by_iris.get(str(row["probe_iris_id"]), {})
        true_rel = true_gallery.get("relative_path", "")

        probe_topk = topk[topk["probe_relative_path"] == probe_rel].copy()
        probe_twins = probe_topk[probe_topk["is_twin"] == True].sort_values("rank")

        if len(probe_twins) == 0:
            continue

        twin_row = probe_twins.iloc[0]
        twin_rel = twin_row["gallery_relative_path"]

        twin_items.append({
            "probe": probe_rel,
            "true_gallery": true_rel,
            "other_gallery": twin_rel,
            "line1": f"twin_rank={int(twin_row['rank'])} twin_score={float(twin_row['score']):.4f}",
            "line2": f"true_rank={row['true_rank']} margin_twin_true={row['margin_best_twin_minus_true']:.4f}",
        })

    make_identification_contact_sheet(
        twin_items,
        CONTACT_DIR / "twin_near_misses.png",
        "Twin near misses: probe / true gallery / closest twin",
        feature_by_rel,
        n=TOP_N,
    )

    # Statistiche e correlazioni
    genuine_all = matching[matching["relation"] == "genuine"].copy()
    unrelated_all = matching[matching["relation"] == "unrelated"].copy()
    twin_same_all = matching[matching["relation"] == "twin_same_eye"].copy()
    twin_cross_all = matching[matching["relation"] == "twin_cross_eye"].copy()

    rank1_error_relation_counts = (
        rank1_errors["best_impostor_relation_to_probe"]
        .value_counts(dropna=False)
        .to_dict()
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
            "features": str(INPUT_FEATURES),
            "matching": str(INPUT_MATCHING),
            "closed_set_results": str(INPUT_CLOSED),
            "closed_set_topk": str(INPUT_TOPK),
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
                genuine_all["valid_bit_ratio_mean"]
            ),
            "unrelated_hd_vs_valid_bit_ratio_mean": corr(
                unrelated_all["hamming_distance"],
                unrelated_all["valid_bit_ratio_mean"]
            ),
            "closed_set_margin_vs_probe_valid_bit_ratio": corr(
                closed["margin_best_impostor_minus_true"],
                closed["probe_valid_bit_ratio"]
            ),
            "closed_set_true_rank_vs_probe_valid_bit_ratio": corr(
                closed["true_rank"],
                closed["probe_valid_bit_ratio"]
            ),
        },
        "outputs": {
            "hard_genuine_pairs_csv": str(OUTPUT_DIR / "hard_genuine_pairs.csv"),
            "low_unrelated_pairs_csv": str(OUTPUT_DIR / "low_unrelated_pairs.csv"),
            "low_twin_same_eye_pairs_csv": str(OUTPUT_DIR / "low_twin_same_eye_pairs.csv"),
            "low_twin_cross_eye_pairs_csv": str(OUTPUT_DIR / "low_twin_cross_eye_pairs.csv"),
            "rank1_errors_csv": str(OUTPUT_DIR / "rank1_errors.csv"),
            "twin_near_misses_csv": str(OUTPUT_DIR / "twin_near_misses.csv"),
            "contact_sheets": {
                "hard_genuine_pairs": str(CONTACT_DIR / "hard_genuine_pairs.png"),
                "low_unrelated_pairs": str(CONTACT_DIR / "low_unrelated_pairs.png"),
                "low_twin_same_eye_pairs": str(CONTACT_DIR / "low_twin_same_eye_pairs.png"),
                "low_twin_cross_eye_pairs": str(CONTACT_DIR / "low_twin_cross_eye_pairs.png"),
                "rank1_errors": str(CONTACT_DIR / "rank1_errors.png"),
                "twin_near_misses": str(CONTACT_DIR / "twin_near_misses.png"),
            }
        }
    }

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
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
    for k, v in rank1_error_relation_counts.items():
        print(f"  {k}: {v}")
    print()
    print("Correlations:")
    for k, v in summary["correlations"].items():
        print(f"  {k}: {v}")
    print()
    print(f"Summary:        {SUMMARY_JSON}")
    print(f"Contact sheets: {CONTACT_DIR}")


if __name__ == "__main__":
    main()