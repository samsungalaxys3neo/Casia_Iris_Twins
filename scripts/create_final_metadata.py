from pathlib import Path
import pandas as pd


QUALITY_REPORT = Path("iris_twins_project/data/quality/quality_report.csv")
OUTPUT_PATH = Path("iris_twins_project/data/metadata/metadata_final.csv")


def main():
    df = pd.read_csv(
        QUALITY_REPORT,
        dtype={
            "family_id": str,
            "twin_id": str,
            "eye": str,
            "image_idx": str,
        }
    )

    # Soglie data-driven: prendiamo l'1% più estremo per ogni metrica.
    focus_q01 = df["focus_score"].quantile(0.01)
    contrast_q01 = df["contrast"].quantile(0.01)
    bright_q99 = df["bright_ratio"].quantile(0.99)
    dark_q99 = df["dark_ratio"].quantile(0.99)

    df["flag_low_focus"] = df["focus_score"] <= focus_q01
    df["flag_low_contrast"] = df["contrast"] <= contrast_q01
    df["flag_high_bright_ratio"] = df["bright_ratio"] >= bright_q99
    df["flag_high_dark_ratio"] = df["dark_ratio"] >= dark_q99

    # Per ora non scartiamo nulla.
    df["quality_reject"] = False

    df["quality_flag_any"] = (
        df["flag_low_focus"]
        | df["flag_low_contrast"]
        | df["flag_high_bright_ratio"]
        | df["flag_high_dark_ratio"]
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("=== FINAL METADATA REPORT ===")
    print(f"Input immagini:  {len(df)}")
    print(f"Output salvato:  {OUTPUT_PATH}")
    print()
    print("=== Soglie usate ===")
    print(f"focus_score <= {focus_q01:.4f}  -> flag_low_focus")
    print(f"contrast <= {contrast_q01:.4f}  -> flag_low_contrast")
    print(f"bright_ratio >= {bright_q99:.4f} -> flag_high_bright_ratio")
    print(f"dark_ratio >= {dark_q99:.4f}    -> flag_high_dark_ratio")
    print()
    print("=== Conteggio flag ===")
    print(f"flag_low_focus:          {int(df['flag_low_focus'].sum())}")
    print(f"flag_low_contrast:       {int(df['flag_low_contrast'].sum())}")
    print(f"flag_high_bright_ratio:  {int(df['flag_high_bright_ratio'].sum())}")
    print(f"flag_high_dark_ratio:    {int(df['flag_high_dark_ratio'].sum())}")
    print(f"quality_flag_any:        {int(df['quality_flag_any'].sum())}")
    print(f"quality_reject:          {int(df['quality_reject'].sum())}")


if __name__ == "__main__":
    main()