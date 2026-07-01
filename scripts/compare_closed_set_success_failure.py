from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FEATURES = PROJECT_ROOT / "data/features/daugman_strict_v3/features_report.csv"
DEFAULT_INPUT_CLOSED = PROJECT_ROOT / "data/identification_closed_set/daugman_strict_v3/closed_set_results.csv"
DEFAULT_INPUT_TOPK = PROJECT_ROOT / "data/identification_closed_set/daugman_strict_v3/closed_set_topk.csv.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/failure_analysis/daugman_strict_v3/success_failure_comparison"

TOP_N = 12
THUMB_W = 220
THUMB_H = 70
LABEL_H = 70


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


def project_relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def fmt(x, digits=4):
    value = safe_float(x)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def read_features(path: Path):
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
        "code_preview_path": str,
    }

    df = pd.read_csv(path, dtype=dtype_map)

    if "feature_ok" in df.columns:
        df = df[as_bool(df["feature_ok"])].copy()

    df["valid_bit_ratio"] = pd.to_numeric(df["valid_bit_ratio"], errors="coerce")

    feature_by_rel = {
        str(row["relative_path"]): row.to_dict()
        for _, row in df.iterrows()
    }

    gallery_by_iris = {}
    for iris_id, group in df.groupby("iris_id"):
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
    Usa prima la masked preview, perché mostra la parte realmente usata
    dal matcher. Se non esiste, usa la normalized image o la code preview.
    """
    row = feature_by_rel.get(str(relative_path))
    if row is None:
        return None

    for col in ["masked_preview_path", "normalized_path", "code_preview_path"]:
        path = valid_path(row.get(col))
        if path is not None:
            return path

    return None


def load_thumb(relative_path, feature_by_rel, size=(THUMB_W, THUMB_H)):
    path = visual_path(relative_path, feature_by_rel)

    if path is None:
        img = Image.new("RGB", size, color=(235, 235, 235))
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), "image not found", fill=(0, 0, 0))
        return img

    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size)
        return img
    except Exception:
        img = Image.new("RGB", size, color=(235, 235, 235))
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), "load error", fill=(0, 0, 0))
        return img


def draw_text(draw, xy, text, font, max_len=62):
    text = str(text)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def make_three_column_sheet(items, output_path, title, feature_by_rel, n=TOP_N):
    if len(items) == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    items = items[:n]

    font = ImageFont.load_default()

    item_w = THUMB_W * 3
    item_h = THUMB_H + LABEL_H
    title_h = 36

    sheet = Image.new(
        "RGB",
        (item_w, title_h + len(items) * item_h),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(sheet)
    draw.text((10, 12), title, fill=(0, 0, 0), font=font)

    y0 = title_h

    for idx, item in enumerate(items):
        x = 0
        y = y0 + idx * item_h

        probe_img = load_thumb(item["probe"], feature_by_rel)
        true_img = load_thumb(item["true_gallery"], feature_by_rel)
        other_img = load_thumb(item["other_gallery"], feature_by_rel)

        sheet.paste(probe_img, (x, y))
        sheet.paste(true_img, (x + THUMB_W, y))
        sheet.paste(other_img, (x + 2 * THUMB_W, y))

        draw_text(draw, (x + 4, y + THUMB_H + 4), f"Probe: {item['probe']}", font)
        draw_text(draw, (x + THUMB_W + 4, y + THUMB_H + 4), f"True: {item['true_gallery']}", font)
        draw_text(draw, (x + 2 * THUMB_W + 4, y + THUMB_H + 4), f"Other: {item['other_gallery']}", font)

        draw_text(draw, (x + 4, y + THUMB_H + 24), item["line1"], font, max_len=110)
        draw_text(draw, (x + 4, y + THUMB_H + 44), item["line2"], font, max_len=110)

    sheet.save(output_path)
    print(f"Saved: {project_display_path(output_path)}")


def read_closed(path: Path):
    df = pd.read_csv(
        path,
        dtype={
            "probe_relative_path": str,
            "probe_iris_id": str,
        },
    )

    df["rank1_correct"] = as_bool(df["rank1_correct"])
    df["true_rank"] = pd.to_numeric(df["true_rank"], errors="coerce")
    df["margin_best_impostor_minus_true"] = pd.to_numeric(
        df["margin_best_impostor_minus_true"],
        errors="coerce",
    )

    if "min_twin_rank" in df.columns:
        df["min_twin_rank"] = pd.to_numeric(df["min_twin_rank"], errors="coerce")
    else:
        df["min_twin_rank"] = np.nan

    return df


def read_topk(path: Path):
    df = pd.read_csv(
        path,
        compression="gzip",
        dtype={
            "probe_relative_path": str,
            "gallery_relative_path": str,
            "probe_iris_id": str,
            "gallery_iris_id": str,
            "relation_to_probe": str,
        },
    )

    for col in ["is_true", "is_twin", "is_twin_same_eye", "is_twin_cross_eye"]:
        if col in df.columns:
            df[col] = as_bool(df[col])

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    return df


def first_topk_candidate(topk, probe_rel, condition=None):
    probe_topk = topk[topk["probe_relative_path"] == probe_rel].copy()

    if condition is not None:
        probe_topk = probe_topk[condition(probe_topk)].copy()

    if len(probe_topk) == 0:
        return None

    return probe_topk.sort_values("rank").iloc[0]


def relation_col(df):
    if "best_impostor_relation_to_probe" in df.columns:
        return "best_impostor_relation_to_probe"
    if "rank1_relation_to_probe" in df.columns:
        return "rank1_relation_to_probe"
    raise ValueError("No relation column found in closed_set_results.csv")


def build_items(rows_df, topk, gallery_by_iris, feature_by_rel, mode):
    items = []

    for _, row in rows_df.iterrows():
        probe_rel = row["probe_relative_path"]
        true_gallery = gallery_by_iris.get(str(row["probe_iris_id"]), {})
        true_rel = true_gallery.get("relative_path", "")

        if mode == "success":
            other = first_topk_candidate(
                topk,
                probe_rel,
                condition=lambda x: x["is_true"] == False,
            )
            if other is None:
                continue

            other_rel = other["gallery_relative_path"]
            relation = other.get("relation_to_probe", "")
            score = other.get("score", np.nan)

            line1 = (
                f"SUCCESS | true_rank={fmt(row['true_rank'], 0)} "
                f"margin={fmt(row['margin_best_impostor_minus_true'])}"
            )
            line2 = (
                f"best_impostor={relation} "
                f"rank={fmt(other['rank'], 0)} score={fmt(score)}"
            )

        elif mode == "failure":
            other = first_topk_candidate(
                topk,
                probe_rel,
                condition=lambda x: x["rank"] == 1,
            )
            if other is None:
                continue

            other_rel = other["gallery_relative_path"]
            relation = other.get("relation_to_probe", "")
            score = other.get("score", np.nan)

            line1 = (
                f"FAILURE | true_rank={fmt(row['true_rank'], 0)} "
                f"margin={fmt(row['margin_best_impostor_minus_true'])}"
            )
            line2 = (
                f"wrong_rank1={relation} "
                f"score={fmt(score)} twin_rank={fmt(row.get('min_twin_rank', np.nan), 0)}"
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        items.append(
            {
                "probe": probe_rel,
                "true_gallery": true_rel,
                "other_gallery": other_rel,
                "line1": line1,
                "line2": line2,
            }
        )

    return items


def run(input_features: Path, input_closed: Path, input_topk: Path, output_dir: Path, top_n: int):
    input_features = resolve_from_project(input_features)
    input_closed = resolve_from_project(input_closed)
    input_topk = resolve_from_project(input_topk)
    output_dir = resolve_from_project(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    features_df, feature_by_rel, gallery_by_iris = read_features(input_features)
    closed = read_closed(input_closed)
    topk = read_topk(input_topk)

    rel_col = relation_col(closed)

    print("=== CLOSED-SET SUCCESS VS FAILURE COMPARISON ===")
    print(f"Features:   {project_display_path(input_features)}")
    print(f"Closed-set: {project_display_path(input_closed)}")
    print(f"Top-k:      {project_display_path(input_topk)}")
    print(f"Output dir: {project_display_path(output_dir)}")
    print()

    # Successi molto sicuri: rank1 corretto e margine alto
    successful = (
        closed[
            (closed["rank1_correct"] == True)
            & (closed["margin_best_impostor_minus_true"] > 0.08)
        ]
        .sort_values("margin_best_impostor_minus_true", ascending=False)
        .head(top_n)
        .copy()
    )

    # Errori veri e forti: rank1 sbagliato e margine negativo marcato
    failures = closed[closed["rank1_correct"] == False].copy()

    unrelated_failures = (
        failures[failures[rel_col] == "unrelated"]
        .sort_values("margin_best_impostor_minus_true", ascending=True)
        .head(top_n)
        .copy()
    )

    twin_failures = (
        failures[failures[rel_col].isin(["twin_same_eye", "twin_cross_eye"])]
        .sort_values("margin_best_impostor_minus_true", ascending=True)
        .head(top_n)
        .copy()
    )

    twin_not_near_failures = (
        failures[
            (~failures[rel_col].isin(["twin_same_eye", "twin_cross_eye"]))
            & (
                failures["min_twin_rank"].isna()
                | (failures["min_twin_rank"] > 20)
            )
        ]
        .sort_values("margin_best_impostor_minus_true", ascending=True)
        .head(top_n)
        .copy()
    )

    successful.to_csv(output_dir / "successful_rank1_high_margin.csv", index=False)
    unrelated_failures.to_csv(output_dir / "rank1_errors_unrelated.csv", index=False)
    twin_failures.to_csv(output_dir / "rank1_errors_twin.csv", index=False)
    twin_not_near_failures.to_csv(output_dir / "rank1_errors_twin_not_near.csv", index=False)

    success_items = build_items(successful, topk, gallery_by_iris, feature_by_rel, mode="success")
    unrelated_items = build_items(unrelated_failures, topk, gallery_by_iris, feature_by_rel, mode="failure")
    twin_items = build_items(twin_failures, topk, gallery_by_iris, feature_by_rel, mode="failure")
    twin_not_near_items = build_items(twin_not_near_failures, topk, gallery_by_iris, feature_by_rel, mode="failure")

    make_three_column_sheet(
        success_items,
        output_dir / "successful_rank1_examples.png",
        "Successful Rank-1 examples: probe / true gallery / best impostor",
        feature_by_rel,
        n=top_n,
    )

    make_three_column_sheet(
        unrelated_items,
        output_dir / "rank1_errors_unrelated.png",
        "Rank-1 errors caused by unrelated impostors: probe / true / wrong rank-1",
        feature_by_rel,
        n=top_n,
    )

    make_three_column_sheet(
        twin_items,
        output_dir / "rank1_errors_twin.png",
        "Rank-1 errors caused by twins: probe / true / wrong twin rank-1",
        feature_by_rel,
        n=top_n,
    )

    make_three_column_sheet(
        twin_not_near_items,
        output_dir / "rank1_errors_twin_not_near.png",
        "Rank-1 errors where twin is not near top ranks: probe / true / wrong rank-1",
        feature_by_rel,
        n=top_n,
    )

    # Un'immagine singola facile per la slide: successi sopra, errori sotto
    make_three_column_sheet(
        success_items[:6] + unrelated_items[:6],
        output_dir / "success_vs_unrelated_failure_examples.png",
        "Success vs failure: successful Rank-1 examples followed by unrelated Rank-1 errors",
        feature_by_rel,
        n=12,
    )

    summary = {
        "inputs": {
            "features": project_display_path(input_features),
            "closed_set_results": project_display_path(input_closed),
            "topk": project_display_path(input_topk),
        },
        "counts": {
            "closed_set_probes": int(len(closed)),
            "rank1_correct": int(closed["rank1_correct"].sum()),
            "rank1_errors": int((closed["rank1_correct"] == False).sum()),
            "successful_high_margin_selected": int(len(successful)),
            "rank1_errors_unrelated_selected": int(len(unrelated_failures)),
            "rank1_errors_twin_selected": int(len(twin_failures)),
            "rank1_errors_twin_not_near_selected": int(len(twin_not_near_failures)),
        },
        "selection_rules": {
            "successful_rank1_high_margin": "rank1_correct == True and margin_best_impostor_minus_true > 0.08",
            "rank1_errors_unrelated": f"rank1_correct == False and {rel_col} == 'unrelated'",
            "rank1_errors_twin": f"rank1_correct == False and {rel_col} in ['twin_same_eye', 'twin_cross_eye']",
            "rank1_errors_twin_not_near": f"rank1_correct == False and not twin-caused and min_twin_rank > 20 or missing",
        },
        "outputs": {
            "successful_rank1_examples": project_display_path(output_dir / "successful_rank1_examples.png"),
            "rank1_errors_unrelated": project_display_path(output_dir / "rank1_errors_unrelated.png"),
            "rank1_errors_twin": project_display_path(output_dir / "rank1_errors_twin.png"),
            "rank1_errors_twin_not_near": project_display_path(output_dir / "rank1_errors_twin_not_near.png"),
            "success_vs_unrelated_failure_examples": project_display_path(output_dir / "success_vs_unrelated_failure_examples.png"),
        },
    }

    with (output_dir / "success_failure_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    print()
    print("=== SUMMARY ===")
    print(f"Closed-set probes:       {summary['counts']['closed_set_probes']}")
    print(f"Rank-1 correct:          {summary['counts']['rank1_correct']}")
    print(f"Rank-1 errors:           {summary['counts']['rank1_errors']}")
    print(f"Selected successes:      {summary['counts']['successful_high_margin_selected']}")
    print(f"Selected unrelated errs: {summary['counts']['rank1_errors_unrelated_selected']}")
    print(f"Selected twin errs:      {summary['counts']['rank1_errors_twin_selected']}")
    print(f"Output dir:              {project_display_path(output_dir)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-features", type=Path, default=DEFAULT_INPUT_FEATURES)
    parser.add_argument("--input-closed", type=Path, default=DEFAULT_INPUT_CLOSED)
    parser.add_argument("--input-topk", type=Path, default=DEFAULT_INPUT_TOPK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N)

    args = parser.parse_args()

    run(
        input_features=args.input_features,
        input_closed=args.input_closed,
        input_topk=args.input_topk,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()