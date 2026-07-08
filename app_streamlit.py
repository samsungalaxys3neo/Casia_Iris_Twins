import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import norm, gaussian_kde
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Behind the Iris Demo",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CONSTANTS
# ============================================================
PAPER_TITLE_LINE_1 = "BEHIND THE IRIS:"
PAPER_TITLE_LINE_2 = "Biometric Recognition in Twin Subjects"
PAPER_TITLE = f"{PAPER_TITLE_LINE_1} {PAPER_TITLE_LINE_2}"
AUTHORS = "Sarah Oualli, Michele Pogu, Gaia Diodati"
FRAMEWORK = "Daugman-style strict pipeline · Segmentation · Rubber Sheet Normalization · Gabor-style Encoding"
RUN_COMMAND = "python3 scripts/demo_orale_strict_sample_v3.py --seed 1234 --families 20 --images-per-iris 5"

FUCHSIA = "#e6007e"
FUCHSIA_DARK = "#b00060"
FUCHSIA_LIGHT = "#fff5fb"
BLACK = "#111111"
WHITE = "#ffffff"
LIGHT_GREY = "#ebebeb"
MID_GREY = "#5e5e5e"

RELATION_ORDER = [
    "genuine",
    "twin_same_eye",
    "twin_cross_eye",
    "same_subject_different_eye",
    "unrelated",
]

RELATION_LABELS = {
    "genuine": "Genuine",
    "twin_same_eye": "Twin same eye",
    "twin_cross_eye": "Twin cross eye",
    "same_subject_different_eye": "Same subject / different eye",
    "unrelated": "Unrelated",
}

RELATION_COLORS = {
    "genuine": "#1f77b4",
    "twin_same_eye": FUCHSIA,
    "twin_cross_eye": "#ff8c00",
    "same_subject_different_eye": "#2ca02c",
    "unrelated": "#7a7a7a",
}

LATEST_COUNTS = pd.DataFrame(
    [
        {"relation": "unrelated", "count": 17258},
        {"relation": "same_subject_different_eye", "count": 316},
        {"relation": "genuine", "count": 290},
        {"relation": "twin_cross_eye", "count": 240},
        {"relation": "twin_same_eye", "count": 232},
    ]
)

LATEST_HD_STATS = pd.DataFrame(
    [
        {
            "relation": "genuine",
            "count": 290,
            "mean": 0.301799,
            "median": 0.298098,
            "std": 0.064129,
            "min": 0.126068,
            "max": 0.445280,
        },
        {
            "relation": "twin_same_eye",
            "count": 232,
            "mean": 0.427954,
            "median": 0.428497,
            "std": 0.025944,
            "min": 0.363787,
            "max": 0.494627,
        },
        {
            "relation": "same_subject_different_eye",
            "count": 316,
            "mean": 0.434096,
            "median": 0.432591,
            "std": 0.024501,
            "min": 0.365162,
            "max": 0.517283,
        },
        {
            "relation": "twin_cross_eye",
            "count": 240,
            "mean": 0.434418,
            "median": 0.436070,
            "std": 0.027227,
            "min": 0.370285,
            "max": 0.515605,
        },
        {
            "relation": "unrelated",
            "count": 797,
            "mean": 0.444877,
            "median": 0.444105,
            "std": 0.023157,
            "min": 0.387895,
            "max": 0.537741,
        },
    ]
)

LATEST_GLOBAL = {
    "genuine_pairs": 290,
    "impostor_pairs": 1585,
    "EER": 0.08270423148047427,
    "EER_threshold": 0.4037403740374037,
    "AUC": 0.9775089742195149,
}

# Gaussian parameters used by the interactive live simulation.
# Genuine parameters come directly from the latest sample-specific statistics.
# Impostor parameters are pooled from all non-genuine relation classes.
LIVE_MU_GENUINE = 0.301799
LIVE_SIGMA_GENUINE = 0.064129
LIVE_MU_IMPOSTOR = 0.438666847318612
LIVE_SIGMA_IMPOSTOR = 0.025337380984370276

# The empirical report EER is 8.27% at HD 0.403740, computed from the real score arrays.
# The live chart below is an analytic Gaussian approximation built only from mean/std.
# Therefore its own FAR=FRR operating point is slightly different.
LIVE_GAUSSIAN_EER = 0.06303110950887554
LIVE_GAUSSIAN_EER_THRESHOLD = 0.3999069990699907
LIVE_INITIAL_THRESHOLD = LIVE_GAUSSIAN_EER_THRESHOLD

# ============================================================
# CSS
# ============================================================
st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {{
        --fuchsia: {FUCHSIA};
        --fuchsia-dark: {FUCHSIA_DARK};
        --fuchsia-light: {FUCHSIA_LIGHT};
        --black: {BLACK};
        --white: {WHITE};
        --light-grey: {LIGHT_GREY};
    }}

    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(230, 0, 126, 0.10), transparent 20%),
            linear-gradient(180deg, #ffffff 0%, #fffafd 42%, #ffffff 100%);
        color: var(--black);
        font-family: 'Outfit', sans-serif;
    }}

    .block-container {{
        padding-top: 2.8rem;
        padding-bottom: 4rem;
        max-width: 1500px;
    }}

    p, li, div, span, label {{
        font-size: 1.50rem;
        line-height: 1.78;
        color: var(--black);
    }}

    strong {{
        color: var(--black);
        font-weight: 800;
    }}

    h1, h2, h3 {{
        font-family: 'Space Grotesk', 'Outfit', sans-serif;
        letter-spacing: -0.03em;
        font-weight: 800;
    }}

    h1 {{
        font-size: 5.15rem !important;
        line-height: 1.02 !important;
        color: var(--black) !important;
        margin-bottom: 0.95rem !important;
        max-width: 100%;
        text-align: center;
    }}

    h2 {{
        font-size: 2.95rem !important;
        line-height: 1.10 !important;
        margin-top: 2.1rem !important;
        margin-bottom: 1rem !important;
        padding: 1.0rem 1.2rem !important;
        background: linear-gradient(90deg, var(--fuchsia), var(--fuchsia-dark));
        color: #ffffff !important;
        border-radius: 20px;
        box-shadow: 0 12px 24px rgba(230, 0, 126, 0.18);
    }}

    h3 {{
        font-size: 2.20rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.65rem !important;
        color: var(--black) !important;
    }}

    hr {{
        border: none;
        height: 2px;
        background: linear-gradient(90deg, var(--fuchsia), rgba(230, 0, 126, 0.04));
        margin: 2rem 0;
    }}

    .hero-card {{
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(230, 0, 126, 0.10), rgba(255, 255, 255, 0.0) 34%),
            linear-gradient(180deg, #ffffff 0%, #fff8fc 100%);
        border: 2px solid rgba(230, 0, 126, 0.24);
        border-top: 22px solid var(--fuchsia);
        border-radius: 32px;
        padding: 3.15rem 3.0rem 2.75rem 3.0rem;
        box-shadow: 0 18px 42px rgba(17, 17, 17, 0.09);
        margin: 2.1rem auto 2.2rem auto;
        max-width: 1320px;
        text-align: center;
    }}

    .hero-card::after {{
        content: "";
        position: absolute;
        right: -60px;
        top: -40px;
        width: 230px;
        height: 230px;
        background: radial-gradient(circle, rgba(230, 0, 126, 0.12) 0%, rgba(230, 0, 126, 0.0) 68%);
        pointer-events: none;
    }}

    .cover-kicker {{
        display: inline-block;
        color: #ffffff;
        background: var(--black);
        border-radius: 999px;
        padding: 0.28rem 0.88rem 0.32rem 0.88rem;
        font-size: 1.10rem;
        letter-spacing: 0.16em;
        font-weight: 900;
        text-transform: uppercase;
        margin-bottom: 0.95rem;
    }}

    .demo-badge {{
        display: inline-block;
        color: #ffffff;
        background: var(--fuchsia);
        border-radius: 999px;
        padding: 0.34rem 1.05rem 0.38rem 1.05rem;
        font-size: 1.30rem;
        letter-spacing: 0.14em;
        font-weight: 900;
        text-transform: uppercase;
        margin: 0 auto 1.05rem auto;
        box-shadow: 0 0 0 7px rgba(230, 0, 126, 0.08);
    }}

    .hero-title {{
        font-family: 'Space Grotesk', 'Outfit', sans-serif;
        color: var(--black);
        text-align: center;
        margin: 0 auto 1.15rem auto;
        width: 100%;
    }}

    .hero-title-main,
    .hero-title-sub {{
        display: block;
        font-family: 'Space Grotesk', 'Outfit', sans-serif !important;
        color: var(--black) !important;
        font-weight: 900 !important;
        letter-spacing: 0.02em !important;
        line-height: 1.22 !important;
        text-align: center !important;
        white-space: nowrap;
        text-rendering: geometricPrecision;
    }}

    .hero-title-main {{
        font-size: min(96px, 7.8vw) !important;
        margin-top: 1.35rem;
        margin-bottom: 0.70rem;
        word-spacing: 0.02em;
    }}

    .hero-title-sub {{
        font-size: min(60px, 4.8vw) !important;
        margin-bottom: 1.45rem;
        word-spacing: 0.02em;
    }}

    @media (max-width: 980px) {{
        .hero-title-main,
        .hero-title-sub {{
            white-space: normal;
        }}
        .hero-title-main {{
            font-size: min(58px, 12vw) !important;
        }}
        .hero-title-sub {{
            font-size: min(44px, 8.8vw) !important;
        }}
    }}

    .authors-line {{
        font-size: 1.62rem;
        font-weight: 700;
        color: var(--black);
        margin-top: 0.65rem;
    }}

    .framework-line {{
        font-size: 1.38rem;
        font-weight: 600;
        color: #454545;
        margin-top: 0.25rem;
    }}

    .lead-text {{
        font-size: 1.58rem;
        line-height: 1.72;
        color: #1f1f1f;
        max-width: 1140px;
        margin: 1.35rem auto 0 auto;
    }}

    .fuchsia-box {{
        background: #fff7fb;
        border: 2px solid rgba(230, 0, 126, 0.20);
        border-left: 10px solid var(--fuchsia);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin: 1.15rem 0 1.35rem 0;
        box-shadow: 0 8px 18px rgba(230, 0, 126, 0.06);
    }}

    .fuchsia-box p, .fuchsia-box li, .fuchsia-box span, .fuchsia-box div {{
        font-size: 1.48rem;
        color: var(--black);
        line-height: 1.72;
    }}

    .mini-title {{
        display: inline-block;
        background: var(--fuchsia);
        color: #ffffff;
        border-radius: 12px;
        padding: 0.32rem 0.72rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
        font-size: 1.08rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    div[data-testid="stMetric"] {{
        background: #ffffff;
        border: 1.5px solid rgba(230, 0, 126, 0.22);
        border-radius: 18px;
        padding: 1.0rem 1.05rem;
        box-shadow: 0 8px 18px rgba(17, 17, 17, 0.05);
    }}

    div[data-testid="stMetricLabel"] p {{
        color: var(--black) !important;
        font-size: 1.26rem !important;
        font-weight: 800 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: var(--fuchsia) !important;
        font-size: 2.95rem !important;
        font-weight: 900 !important;
    }}

    div[data-testid="stMetricDelta"] {{
        font-size: 1rem !important;
    }}

    .stAlert {{
        background: #fff7fb !important;
        border: 1.5px solid rgba(230, 0, 126, 0.30) !important;
        border-left: 10px solid var(--fuchsia) !important;
        border-radius: 18px !important;
        color: var(--black) !important;
    }}

    .stAlert p, .stAlert div, .stAlert span {{
        color: var(--black) !important;
        font-size: 1.42rem !important;
        line-height: 1.68 !important;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1.5px solid #efefef;
        border-radius: 16px;
        overflow: hidden;
    }}

    div[data-testid="stImageCaption"] {{
        font-size: 1.38rem !important;
        color: #333333 !important;
        font-weight: 600;
        text-align: center;
    }}

    div.stSlider label p {{
        font-size: 1.38rem !important;
        font-weight: 800 !important;
        color: var(--black) !important;
    }}

    div.stSlider > div[data-baseweb="slider"] > div > div > div {{
        background-color: var(--fuchsia) !important;
    }}

    .custom-table {{
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(17, 17, 17, 0.055);
        border: 1.5px solid #eeeeee;
        margin-bottom: 1.2rem;
    }}

    .custom-table th {{
        background: linear-gradient(90deg, var(--fuchsia), var(--fuchsia-dark));
        color: #ffffff;
        text-align: left;
        font-size: 1.20rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.78rem 0.9rem;
        font-weight: 900;
    }}

    .custom-table td {{
        color: var(--black);
        font-size: 1.38rem;
        padding: 0.78rem 0.9rem;
        border-bottom: 1px solid #eeeeee;
        font-weight: 600;
    }}

    .custom-table tr:nth-child(even) td {{
        background: #fff8fc;
    }}

    .live-panel {{
        background: #ffffff;
        border: 1.5px solid rgba(230, 0, 126, 0.18);
        border-radius: 22px;
        padding: 1rem 1rem 0.6rem 1rem;
        box-shadow: 0 8px 22px rgba(17, 17, 17, 0.05);
    }}

    .status-good {{
        text-align: center;
        border: 2px solid #14975a;
        background: #f2fff8;
        color: #0a6b3b;
        border-radius: 18px;
        padding: 1.15rem;
        font-size: 1.70rem;
        font-weight: 900;
    }}

    .status-risk {{
        text-align: center;
        border: 2px solid var(--fuchsia);
        background: #fff7fb;
        color: var(--fuchsia);
        border-radius: 18px;
        padding: 1.15rem;
        font-size: 1.70rem;
        font-weight: 900;
    }}

    .section-text, .section-text p {{
        font-size: 1.56rem !important;
        line-height: 1.76 !important;
        color: var(--black) !important;
    }}

    .bash-box {{
        background: #ffffff;
        color: #111111;
        border: 2px solid #d9d9d9;
        border-radius: 16px;
        padding: 1.05rem 1.2rem;
        margin: 0.65rem 0 1rem 0;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
        font-size: 1.34rem;
        line-height: 1.55;
        box-shadow: 0 6px 16px rgba(17, 17, 17, 0.05);
        white-space: pre-wrap;
    }}

    div[data-testid="stExpander"] {{
        background: #ffffff;
        border: 1.5px solid rgba(230, 0, 126, 0.22);
        border-radius: 18px;
        margin-bottom: 0.9rem;
        box-shadow: 0 8px 20px rgba(17, 17, 17, 0.045);
    }}

    div[data-testid="stExpander"] summary p {{
        font-size: 1.42rem !important;
        font-weight: 900 !important;
        color: var(--black) !important;
    }}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# MATPLOTLIB STYLE
# ============================================================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "axes.edgecolor": "#d9d9d9",
        "axes.labelcolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "text.color": BLACK,
        "axes.titlecolor": BLACK,
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }
)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_real_data():
    path = Path("demo_orale_strict_sample_v3/tables/sample_matching_results.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "hd" in df.columns:
        df = df.dropna(subset=["hd"]).copy()
    return df


@st.cache_data
def get_image_paths():
    base_dir = Path("demo_orale_strict_sample_v3")
    images = {}
    if base_dir.exists():
        ref_dir = base_dir / "reference_images"
        if ref_dir.exists():
            images = {
                "Raw selected images": ref_dir / "01_raw_selected.png",
                "Segmentation overlays": ref_dir / "02_segmentation_true_overlays.png",
                "Normalized strips": ref_dir / "03_normalized_strips.png",
                "Conservative masks": ref_dir / "04_masks.png",
                "Masked normalized strips": ref_dir / "05_masked_normalized_strips.png",
                "Gabor code previews": ref_dir / "06_gabor_codes.png",
            }
    return {k: v for k, v in images.items() if v.exists()}

# ============================================================
# HELPERS
# ============================================================
def present_relations(df: pd.DataFrame) -> list[str]:
    return [r for r in RELATION_ORDER if r in set(df["relation"].dropna().unique())]


def rel_label(rel: str) -> str:
    return RELATION_LABELS.get(rel, rel.replace("_", " ").title())


def format_relation_df(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "relation" in work.columns:
        work["relation"] = work["relation"].map(lambda x: rel_label(str(x)))
    return work


def render_table_html(df: pd.DataFrame) -> str:
    """Render a large, slide-like HTML table for Streamlit."""
    return df.to_html(index=False, escape=False, classes="custom-table")


def apply_chart_style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=22, fontweight="bold", pad=14, color=BLACK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=18, fontweight="bold", labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18, fontweight="bold", labelpad=10)
    ax.grid(True, color=LIGHT_GREY, linewidth=1.0, alpha=0.95)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#d6d6d6")
        spine.set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=14, colors=BLACK)


def add_legend(ax, loc="best"):
    legend = ax.legend(loc=loc, frameon=True, facecolor=WHITE, edgecolor="#dddddd", framealpha=1)
    if legend:
        for text in legend.get_texts():
            text.set_color(BLACK)
            text.set_fontsize(13)
    return legend


def compute_error_curves(df: pd.DataFrame):
    if "relation" not in df.columns or "hd" not in df.columns:
        return None
    genuine = df.loc[df["relation"] == "genuine", "hd"].dropna().to_numpy()
    impostor = df.loc[df["relation"] != "genuine", "hd"].dropna().to_numpy()
    if len(genuine) == 0 or len(impostor) == 0:
        return None
    min_hd = float(df["hd"].min())
    max_hd = float(df["hd"].max())
    margin = max(0.01, (max_hd - min_hd) * 0.05)
    thresholds = np.linspace(min_hd - margin, max_hd + margin, 500)
    tpr = np.array([(genuine <= t).mean() for t in thresholds])
    fpr = np.array([(impostor <= t).mean() for t in thresholds])
    fnr = 1.0 - tpr
    idx_eer = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)
    eer_threshold = float(thresholds[idx_eer])
    sort_idx = np.argsort(fpr)
    auc = float(np.trapz(tpr[sort_idx], fpr[sort_idx]))
    return {
        "thresholds": thresholds,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc": auc,
    }


def calculate_metrics(threshold, mu_g, sig_g, mu_i, sig_i):
    frr = (1 - norm.cdf(threshold, mu_g, sig_g)) * 100
    far = norm.cdf(threshold, mu_i, sig_i) * 100
    return far, frr


def compute_normal_eer(mu_g, sig_g, mu_i, sig_i):
    """Fallback only: compute EER of a Gaussian model from FAR = FRR."""
    lo = min(mu_g - 5 * sig_g, mu_i - 5 * sig_i, 0.0)
    hi = max(mu_g + 5 * sig_g, mu_i + 5 * sig_i, 0.65)
    thresholds = np.linspace(lo, hi, 2000)
    far = norm.cdf(thresholds, mu_i, sig_i)
    frr = 1.0 - norm.cdf(thresholds, mu_g, sig_g)
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0), float(thresholds[idx])


def score_arrays_from_real_data(df: pd.DataFrame | None, stats: pd.DataFrame):
    """Return real HD arrays when available, otherwise build a deterministic fallback from summary stats."""
    if df is not None and {"relation", "hd"}.issubset(set(df.columns)) and not df.empty:
        work = df.dropna(subset=["relation", "hd"]).copy()
        genuine = work.loc[work["relation"] == "genuine", "hd"].astype(float).to_numpy()
        impostor = work.loc[work["relation"] != "genuine", "hd"].astype(float).to_numpy()
        if len(genuine) >= 2 and len(impostor) >= 2:
            return genuine, impostor

    # Fallback: deterministic synthetic arrays matching the summary means/stds.
    rng = np.random.default_rng(123)
    work = stats.copy()
    genuine_row = work[work["relation"] == "genuine"].iloc[0]
    impostor_rows = work[work["relation"] != "genuine"]
    g_count = int(genuine_row.get("count", 192))
    g_mean = float(genuine_row.get("mean", 0.3105))
    g_std = float(genuine_row.get("std", 0.064))
    genuine = rng.normal(g_mean, max(g_std, 0.01), g_count)
    impostor_parts = []
    for _, row in impostor_rows.iterrows():
        count = int(row.get("count", 100))
        mean = float(row.get("mean", 0.44))
        std = float(row.get("std", 0.025))
        impostor_parts.append(rng.normal(mean, max(std, 0.008), count))
    impostor = np.concatenate(impostor_parts) if impostor_parts else rng.normal(0.44, 0.025, 1000)
    return np.clip(genuine, 0.0, 1.0), np.clip(impostor, 0.0, 1.0)


def apply_noise_to_scores(genuine: np.ndarray, impostor: np.ndarray, noise: float):
    """Deterministic stress transformation: genuine scores become broader and shift right."""
    noise = float(noise)
    g = np.asarray(genuine, dtype=float)
    i = np.asarray(impostor, dtype=float)
    mu_g = float(np.mean(g))
    mu_i = float(np.mean(i))

    stressed_g = mu_g + (g - mu_g) * (1.0 + 0.65 * noise) + 0.040 * noise
    stressed_i = mu_i + (i - mu_i) * (1.0 + 0.20 * noise) - 0.006 * noise
    return np.clip(stressed_g, 0.0, 1.0), np.clip(stressed_i, 0.0, 1.0)


def compute_eer_from_score_arrays(genuine: np.ndarray, impostor: np.ndarray):
    """Compute the EER operating point from FAR(t) and FRR(t).

    Lower Hamming Distance means stronger similarity, therefore a pair is
    accepted when HD <= threshold. The returned threshold is the point where
    FAR and FRR are equal by linear interpolation over the empirical error
    curves. With finite samples, exact equality may not exist; interpolation
    avoids confusing the EER with the visual crossing of the two score-density
    curves.
    """
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    genuine = genuine[np.isfinite(genuine)]
    impostor = impostor[np.isfinite(impostor)]
    if len(genuine) == 0 or len(impostor) == 0:
        return float("nan"), float("nan")

    lo = float(min(np.min(genuine), np.min(impostor)))
    hi = float(max(np.max(genuine), np.max(impostor)))
    margin = max(0.002, (hi - lo) * 0.03)
    thresholds = np.linspace(lo - margin, hi + margin, 6000)
    far = np.array([(impostor <= t).mean() for t in thresholds], dtype=float)
    frr = np.array([(genuine > t).mean() for t in thresholds], dtype=float)
    diff = far - frr

    # Prefer an interpolated zero-crossing of FAR-FRR.
    crossing = np.where(np.signbit(diff[:-1]) != np.signbit(diff[1:]))[0]
    if len(crossing):
        # Choose the crossing closest to the point where the absolute error gap is minimal.
        closest = int(np.argmin(np.abs(diff)))
        idx = min(crossing, key=lambda j: abs(j - closest))
        x0, x1 = thresholds[idx], thresholds[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        alpha = 0.0 if abs(d1 - d0) < 1e-12 else float(-d0 / (d1 - d0))
        alpha = min(max(alpha, 0.0), 1.0)
        threshold = float(x0 + alpha * (x1 - x0))
        far_eq = float(far[idx] + alpha * (far[idx + 1] - far[idx]))
        frr_eq = float(frr[idx] + alpha * (frr[idx + 1] - frr[idx]))
        eer = float((far_eq + frr_eq) / 2.0)
        return eer, threshold

    # Fallback: finite empirical curves may never cross in degenerate cases.
    idx = int(np.argmin(np.abs(diff)))
    return float((far[idx] + frr[idx]) / 2.0), float(thresholds[idx])


def compute_visual_equivalent_eer(genuine: np.ndarray, impostor: np.ndarray):
    """Return the overlap point used by the live plot.

    This is intentionally computed on the same smoothed score densities drawn in
    the chart, so the vertical line matches the visual crossing instead of being
    shifted to a discrete CSV threshold.
    """
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    genuine = genuine[np.isfinite(genuine)]
    impostor = impostor[np.isfinite(impostor)]
    if len(genuine) == 0 or len(impostor) == 0:
        return float("nan"), float("nan")

    lo = float(min(np.percentile(genuine, 0.5), np.percentile(impostor, 0.5)))
    hi = float(max(np.percentile(genuine, 99.5), np.percentile(impostor, 99.5)))
    margin = max(0.02, (hi - lo) * 0.10)
    x = np.linspace(max(0.0, lo - margin), min(0.75, hi + margin), 2400)
    yg = smooth_density(genuine, x)
    yi = smooth_density(impostor, x)

    mu_g = float(np.mean(genuine))
    mu_i = float(np.mean(impostor))
    left, right = sorted([mu_g, mu_i])
    mid_mask = (x >= left) & (x <= right)
    density_mask = (yg > 0.01 * np.nanmax(yg)) & (yi > 0.01 * np.nanmax(yi))
    valid = mid_mask & density_mask
    if not np.any(valid):
        valid = mid_mask
    if not np.any(valid):
        valid = np.ones_like(x, dtype=bool)

    diff = yg - yi
    xv = x[valid]
    dv = diff[valid]
    change = np.where(np.signbit(dv[:-1]) != np.signbit(dv[1:]))[0]
    if len(change):
        # Choose the crossing closest to the center of the two score clouds.
        center = (mu_g + mu_i) / 2.0
        candidates = []
        for idx in change:
            x0, x1 = xv[idx], xv[idx + 1]
            y0, y1 = dv[idx], dv[idx + 1]
            root = x0 if abs(y1 - y0) < 1e-12 else x0 - y0 * (x1 - x0) / (y1 - y0)
            candidates.append(float(root))
        threshold = min(candidates, key=lambda z: abs(z - center))
    else:
        threshold = float(xv[int(np.argmin(np.abs(dv)))])

    far = float((impostor <= threshold).mean())
    frr = float((genuine > threshold).mean())
    equivalent_eer = float((far + frr) / 2.0)
    return equivalent_eer, threshold


def compute_far_frr_from_arrays(threshold: float, genuine: np.ndarray, impostor: np.ndarray):
    far = float((np.asarray(impostor, dtype=float) <= threshold).mean()) * 100.0
    frr = float((np.asarray(genuine, dtype=float) > threshold).mean()) * 100.0
    return far, frr


def smooth_density(values: np.ndarray, x: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) >= 3 and np.std(values) > 1e-6:
        try:
            return gaussian_kde(values)(x)
        except Exception:
            pass
    return norm.pdf(x, float(np.mean(values)), max(float(np.std(values)), 0.01))


def calibrated_live_base_parameters(stats: pd.DataFrame, global_metrics: dict):
    """Fallback parameters for cases where the real matching CSV is not available."""
    target_threshold = float(global_metrics.get("EER_threshold", LATEST_GLOBAL["EER_threshold"]))
    target_eer = float(global_metrics.get("EER", LATEST_GLOBAL["EER"]))
    target_eer = min(max(target_eer, 1e-4), 0.49)
    z = float(norm.ppf(1.0 - target_eer))

    if "relation" not in stats.columns or "mean" not in stats.columns:
        return 0.3105, 0.079, 0.4415, 0.032

    work = stats.copy()
    work["mean"] = pd.to_numeric(work["mean"], errors="coerce")
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(1.0) if "count" in work.columns else 1.0
    genuine_stats = work[work["relation"] == "genuine"].dropna(subset=["mean"])
    impostor_stats = work[work["relation"] != "genuine"].dropna(subset=["mean"])
    mu_g = float(genuine_stats.iloc[0]["mean"]) if not genuine_stats.empty else 0.3105
    mu_i = float(np.average(impostor_stats["mean"], weights=impostor_stats["count"])) if not impostor_stats.empty else 0.4415
    if not (mu_g < target_threshold < mu_i) or z <= 0:
        mu_g, mu_i = 0.3105, 0.4415
    sigma_g = min(max(float((target_threshold - mu_g) / z), 0.020), 0.120)
    sigma_i = min(max(float((mu_i - target_threshold) / z), 0.012), 0.080)
    return float(mu_g), float(sigma_g), float(mu_i), float(sigma_i)

def build_display_summaries(df):
    """Use the generated CSV when available; otherwise fall back to the latest reported run."""
    counts = LATEST_COUNTS.copy()
    stats = LATEST_HD_STATS.copy()
    metrics = dict(LATEST_GLOBAL)
    curves = None

    if df is not None and {"relation", "hd"}.issubset(set(df.columns)) and not df.empty:
        work = df.dropna(subset=["relation", "hd"]).copy()
        counts = (
            work["relation"]
            .value_counts()
            .rename_axis("relation")
            .reset_index(name="count")
        )
        counts["_order"] = counts["relation"].map({r: i for i, r in enumerate(RELATION_ORDER)}).fillna(999)
        counts = counts.sort_values(["_order", "relation"]).drop(columns="_order").reset_index(drop=True)

        stats = (
            work.groupby("relation")["hd"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )
        stats["_order"] = stats["relation"].map({r: i for i, r in enumerate(RELATION_ORDER)}).fillna(999)
        stats = stats.sort_values(["_order", "relation"]).drop(columns="_order").reset_index(drop=True)

        curves = compute_error_curves(work)
        metrics = {
            "genuine_pairs": int((work["relation"] == "genuine").sum()),
            "impostor_pairs": int((work["relation"] != "genuine").sum()),
            "EER": float(curves["eer"]) if curves else float(LATEST_GLOBAL["EER"]),
            "EER_threshold": float(curves["eer_threshold"]) if curves else float(LATEST_GLOBAL["EER_threshold"]),
            "AUC": float(curves["auc"]) if curves else float(LATEST_GLOBAL["AUC"]),
        }

    return counts, stats, metrics, curves


def weighted_impostor_parameters(stats: pd.DataFrame) -> tuple[float, float]:
    impostor = stats[stats["relation"] != "genuine"].copy()
    if impostor.empty:
        return 0.44, 0.025
    weights = impostor["count"].astype(float).clip(lower=1)
    mean = float(np.average(impostor["mean"].astype(float), weights=weights))
    std = float(np.average(impostor["std"].fillna(0.025).astype(float), weights=weights))
    return mean, max(std, 0.01)


def round_to_slider_step(value: float, step: float = 0.005, lo: float = 0.20, hi: float = 0.50) -> float:
    value = min(max(float(value), lo), hi)
    return round(round(value / step) * step, 3)

# ============================================================
# PLOT FUNCTIONS
# ============================================================
def plot_boxplot(df: pd.DataFrame):
    rels = present_relations(df)
    fig, ax = plt.subplots(figsize=(10.8, 6.8), facecolor=WHITE)
    palette = [RELATION_COLORS.get(r, "#777777") for r in rels]
    sns.boxplot(
        data=df,
        x="relation",
        y="hd",
        order=rels,
        showmeans=True,
        width=0.62,
        linewidth=1.45,
        palette=palette,
        ax=ax,
    )
    ax.set_facecolor(WHITE)
    ax.set_xticklabels([rel_label(r) for r in rels], rotation=24, ha="right")
    apply_chart_style(ax, "Boxplot of Hamming Distance by Pair Type", "Pair relation", "Masked Hamming Distance")
    fig.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame):
    rels = present_relations(df)
    work = df[df["relation"].isin(rels)].copy()
    rng = np.random.default_rng(7)
    rel_to_x = {rel: idx for idx, rel in enumerate(rels)}
    work["xpos"] = work["relation"].map(rel_to_x).astype(float) + rng.normal(0, 0.06, len(work))
    fig, ax = plt.subplots(figsize=(11.2, 6.8), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    for rel in rels:
        sub = work[work["relation"] == rel]
        ax.scatter(
            sub["xpos"], sub["hd"], s=44, alpha=0.72,
            color=RELATION_COLORS.get(rel, "#777777"),
            label=f"{rel_label(rel)} (n={len(sub)})",
            edgecolor="white", linewidth=0.35,
        )
    ax.set_xticks(range(len(rels)))
    ax.set_xticklabels([rel_label(r) for r in rels], rotation=24, ha="right")
    apply_chart_style(ax, "Scatter Plot of Pairwise Matching Scores", "Pair relation", "Masked Hamming Distance")
    add_legend(ax, loc="upper left")
    fig.tight_layout()
    return fig


def plot_roc(df: pd.DataFrame):
    curves = compute_error_curves(df)
    if curves is None:
        return None
    fig, ax = plt.subplots(figsize=(10.2, 6.8), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.plot(curves["fpr"], curves["tpr"], color=FUCHSIA, lw=3.0, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#bbbbbb", lw=1.7, label="Random baseline")
    eer = curves["eer"]
    eer_t = curves["eer_threshold"]
    ax.scatter([eer], [1 - eer], color=BLACK, s=85, zorder=5, label=f"EER ≈ {eer*100:.2f}% @ HD {eer_t:.3f}")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    apply_chart_style(ax, "ROC Curve: Genuine vs Impostor Decision", "False Acceptance Rate / FPR", "True Acceptance Rate / TPR")
    add_legend(ax, loc="lower right")
    fig.tight_layout()
    return fig


def plot_det(df: pd.DataFrame):
    curves = compute_error_curves(df)
    if curves is None:
        return None
    eps = 1e-4
    fpr = np.clip(curves["fpr"], eps, 1)
    fnr = np.clip(curves["fnr"], eps, 1)
    fig, ax = plt.subplots(figsize=(10.2, 6.8), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.plot(fpr * 100, fnr * 100, color=FUCHSIA, lw=3.0, label="DET curve")
    ax.scatter([curves["eer"] * 100], [curves["eer"] * 100], color=BLACK, s=85, zorder=5, label=f"EER ≈ {curves['eer']*100:.2f}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(eps * 100, fpr.min() * 80), 100)
    ax.set_ylim(max(eps * 100, fnr.min() * 80), 100)
    apply_chart_style(ax, "DET Curve: FAR vs FRR Trade-off", "False Acceptance Rate / FAR (%)", "False Rejection Rate / FRR (%)")
    add_legend(ax, loc="upper right")
    fig.tight_layout()
    return fig



@st.cache_resource(
    show_spinner=False,
    hash_funcs={
        pd.DataFrame: lambda df: (
            df.shape,
            tuple(df.columns),
            tuple(df["relation"].value_counts().sort_index().items()) if "relation" in df.columns else (),
            round(float(df["hd"].min()), 6) if "hd" in df.columns and len(df) else 0.0,
            round(float(df["hd"].max()), 6) if "hd" in df.columns and len(df) else 0.0,
        )
    },
)
def get_static_figures(df: pd.DataFrame):
    """Cache all static Matplotlib figures so slider changes do not redraw them."""
    return {
        "roc": plot_roc(df),
        "det": plot_det(df),
        "scatter": plot_scatter(df),
        "boxplot": plot_boxplot(df),
    }


def plot_live_distribution_plotly(threshold, noise):
    """Live Gaussian simulation using the latest sample-specific demo metrics."""
    mu_genuine = LIVE_MU_GENUINE
    sigma_genuine = LIVE_SIGMA_GENUINE
    mu_impostor = LIVE_MU_IMPOSTOR
    sigma_impostor = LIVE_SIGMA_IMPOSTOR

    # Noise broadens both distributions and shifts genuine scores rightward.
    new_sigma_g = sigma_genuine + (noise * 0.03)
    new_sigma_i = sigma_impostor + (noise * 0.02)
    new_mu_g = mu_genuine + (noise * 0.04)
    new_mu_i = mu_impostor

    x = np.linspace(0.10, 0.56, 500)
    y_g = norm.pdf(x, new_mu_g, new_sigma_g)
    y_i = norm.pdf(x, new_mu_i, new_sigma_i)
    far, frr = calculate_metrics(threshold, new_mu_g, new_sigma_g, new_mu_i, new_sigma_i)
    gaussian_eer, gaussian_eer_threshold = compute_normal_eer(new_mu_g, new_sigma_g, new_mu_i, new_sigma_i)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y_g,
        mode="lines",
        name="Genuine (true matches)",
        line=dict(color="#1f77b4", width=3),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.26)",
        hovertemplate="HD=%{x:.3f}<br>Density=%{y:.3f}<extra>Genuine</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y_i,
        mode="lines",
        name="Impostors / twins",
        line=dict(color=FUCHSIA, width=3),
        fill="tozeroy",
        fillcolor="rgba(230, 0, 126, 0.22)",
        hovertemplate="HD=%{x:.3f}<br>Density=%{y:.3f}<extra>Impostor</extra>",
    ))

    fig.add_vline(
        x=gaussian_eer_threshold,
        line_width=2.2,
        line_dash="dot",
        line_color=FUCHSIA,
        annotation_text=f"Gaussian EER = {gaussian_eer*100:.2f}% @ HD {gaussian_eer_threshold:.3f}",
        annotation_position="top left",
    )

    fig.add_vline(
        x=threshold,
        line_width=2.8,
        line_dash="dash",
        line_color=BLACK,
        annotation_text=f"Decision threshold = {threshold:.3f}",
        annotation_position="top right",
    )

    fig.add_annotation(
        x=0.17,
        y=17.0,
        xref="x",
        yref="y",
        text=f"FAR: {far:.2f}%<br>FRR: {frr:.2f}%",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(17,17,17,0.25)",
        borderwidth=1,
        borderpad=8,
        font=dict(size=16, color=BLACK),
    )

    fig.update_layout(
        title=dict(text="Live Biometric Analysis: Impact of Noise and Threshold", font=dict(size=28, color=BLACK)),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        hovermode=False,
        uirevision="live",
        margin=dict(l=40, r=28, t=72, b=48),
        legend=dict(bgcolor="rgba(255,255,255,0.90)", bordercolor="#dddddd", borderwidth=1, font=dict(size=15, color=BLACK)),
        xaxis=dict(
            title="Hamming Distance (HD)",
            range=[0.10, 0.56],
            showgrid=True,
            gridcolor=LIGHT_GREY,
            zeroline=False,
            tickfont=dict(size=16, color=BLACK),
            title_font=dict(size=20, color=BLACK),
        ),
        yaxis=dict(
            title="Probability density",
            range=[0, 20],
            showgrid=True,
            gridcolor=LIGHT_GREY,
            zeroline=False,
            tickfont=dict(size=16, color=BLACK),
            title_font=dict(size=20, color=BLACK),
        ),
    )
    return fig, far, frr



# ============================================================
# REAL-DATA LIVE SIMULATION
# ============================================================
@st.cache_data(show_spinner=False)
def get_real_live_arrays(df: pd.DataFrame):
    """Return real genuine/impostor HD arrays plus deterministic perturbation vectors."""
    if df is None or not {"relation", "hd"}.issubset(set(df.columns)):
        return None
    work = df.dropna(subset=["relation", "hd"]).copy()
    genuine = pd.to_numeric(work.loc[work["relation"] == "genuine", "hd"], errors="coerce").dropna().to_numpy(dtype=float)
    impostor = pd.to_numeric(work.loc[work["relation"] != "genuine", "hd"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(genuine) == 0 or len(impostor) == 0:
        return None

    rng = np.random.default_rng(20260708)
    genuine_jitter = rng.normal(0.0, 1.0, len(genuine))
    impostor_jitter = rng.normal(0.0, 1.0, len(impostor))
    return genuine, impostor, genuine_jitter, impostor_jitter


def stress_real_scores(genuine: np.ndarray, impostor: np.ndarray, genuine_jitter: np.ndarray, impostor_jitter: np.ndarray, noise: float):
    """Stress the real demo scores without replacing them by a Gaussian model.

    At noise = 0 the returned arrays are exactly the real CSV scores. Increasing
    noise makes genuine scores less stable by shifting them right and broadening
    them; impostor scores are mainly broadened, with no strong mean shift.
    """
    noise = float(noise)
    genuine_live = genuine + (noise * 0.040) + (genuine_jitter * noise * 0.015)
    impostor_live = impostor + (impostor_jitter * noise * 0.010)
    return np.clip(genuine_live, 0.0, 1.0), np.clip(impostor_live, 0.0, 1.0)


def density_from_real_scores(values: np.ndarray, x: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) >= 3 and float(np.std(values)) > 1e-8:
        try:
            return gaussian_kde(values)(x)
        except Exception:
            pass
    sigma = max(float(np.std(values)), 0.008)
    return norm.pdf(x, float(np.mean(values)), sigma)


def plot_live_real_distribution_plotly(threshold: float, noise: float, live_arrays):
    genuine, impostor, genuine_jitter, impostor_jitter = live_arrays
    genuine_live, impostor_live = stress_real_scores(genuine, impostor, genuine_jitter, impostor_jitter, noise)

    lo = float(min(np.percentile(genuine_live, 0.5), np.percentile(impostor_live, 0.5)))
    hi = float(max(np.percentile(genuine_live, 99.5), np.percentile(impostor_live, 99.5)))
    margin = max(0.035, (hi - lo) * 0.12)
    x_min = max(0.05, lo - margin)
    x_max = min(0.70, hi + margin)
    x = np.linspace(x_min, x_max, 600)

    y_g = density_from_real_scores(genuine_live, x)
    y_i = density_from_real_scores(impostor_live, x)
    y_max = float(max(np.nanmax(y_g), np.nanmax(y_i))) * 1.18

    far, frr = compute_far_frr_from_arrays(threshold, genuine_live, impostor_live)
    eer, eer_threshold = compute_eer_from_score_arrays(genuine_live, impostor_live)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y_g,
        mode="lines",
        name=f"Real genuine scores (n={len(genuine_live)})",
        line=dict(color="#1f77b4", width=3),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.26)",
        hovertemplate="HD=%{x:.3f}<br>Density=%{y:.3f}<extra>Genuine</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y_i,
        mode="lines",
        name=f"Real impostor/twin scores (n={len(impostor_live)})",
        line=dict(color=FUCHSIA, width=3),
        fill="tozeroy",
        fillcolor="rgba(230, 0, 126, 0.22)",
        hovertemplate="HD=%{x:.3f}<br>Density=%{y:.3f}<extra>Impostor/twin</extra>",
    ))

    fig.add_vline(
        x=eer_threshold,
        line_width=2.2,
        line_dash="dot",
        line_color=FUCHSIA,
        annotation_text=f"Empirical EER = {eer*100:.2f}% @ HD {eer_threshold:.3f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=threshold,
        line_width=2.8,
        line_dash="dash",
        line_color=BLACK,
        annotation_text=f"Decision threshold = {threshold:.3f}",
        annotation_position="top right",
    )

    fig.add_annotation(
        x=x_min + (x_max - x_min) * 0.05,
        y=y_max * 0.82,
        xref="x",
        yref="y",
        text=f"FAR: {far:.2f}%<br>FRR: {frr:.2f}%<br>Noise: {noise:.2f}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(17,17,17,0.25)",
        borderwidth=1,
        borderpad=8,
        font=dict(size=17, color=BLACK),
    )

    fig.update_layout(
        title=dict(text="Live Real-Score Simulation: Threshold and Noise", font=dict(size=30, color=BLACK)),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        hovermode=False,
        uirevision="real-live",
        margin=dict(l=42, r=30, t=76, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.90)", bordercolor="#dddddd", borderwidth=1, font=dict(size=15, color=BLACK)),
        xaxis=dict(
            title="Hamming Distance (HD)",
            range=[x_min, x_max],
            showgrid=True,
            gridcolor=LIGHT_GREY,
            zeroline=False,
            tickfont=dict(size=16, color=BLACK),
            title_font=dict(size=20, color=BLACK),
        ),
        yaxis=dict(
            title="Smoothed empirical density",
            range=[0, y_max],
            showgrid=True,
            gridcolor=LIGHT_GREY,
            zeroline=False,
            tickfont=dict(size=16, color=BLACK),
            title_font=dict(size=20, color=BLACK),
        ),
    )
    return fig, far, frr, eer, eer_threshold, len(genuine_live), len(impostor_live)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
<div class="hero-card">
    <div class="demo-badge">DEMO</div>
    <div class="hero-title"><div class="hero-title-main">{PAPER_TITLE_LINE_1}</div><div class="hero-title-sub">{PAPER_TITLE_LINE_2}</div></div>
    <div class="authors-line"><b>Authors:</b> {AUTHORS}</div>
    <div class="framework-line">{FRAMEWORK}</div>
    <p class="lead-text">
        This interactive demo summarizes the biometric evidence of our paper.
        The central question is whether twins, despite sharing the same DNA, produce iris patterns similar enough to challenge a classical iris-recognition pipeline.
        The analysis is presented through strict segmentation, conservative masking, and Hamming-distance matching.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

DF_REAL = load_real_data()
PIPELINE_IMAGES = get_image_paths()
HAS_REAL_DATA = DF_REAL is not None and {"relation", "hd"}.issubset(set(DF_REAL.columns))
DISPLAY_COUNTS, DISPLAY_HD_STATS, DISPLAY_GLOBAL, DISPLAY_CURVES = build_display_summaries(DF_REAL if HAS_REAL_DATA else None)

# ============================================================
# SECTION 1
# ============================================================
st.header("1. Latest Run Summary")
st.markdown(
    f"""
<div class="section-text">
    This section reports the latest demo run generated with:
</div>
<div class="bash-box">{RUN_COMMAND}</div>
<div class="section-text">
    The values below summarize the key verification outcomes and relation-level statistics.
</div>
""",
    unsafe_allow_html=True,
)

metric_row_1 = st.columns(5)
metric_row_1[0].metric("Genuine pairs", f"{DISPLAY_GLOBAL['genuine_pairs']}")
metric_row_1[1].metric("Impostor pairs", f"{DISPLAY_GLOBAL['impostor_pairs']}")
metric_row_1[2].metric("EER", f"{DISPLAY_GLOBAL['EER']*100:.2f}%")
metric_row_1[3].metric("EER threshold", f"{DISPLAY_GLOBAL['EER_threshold']:.3f}")
metric_row_1[4].metric("AUC", f"{DISPLAY_GLOBAL['AUC']:.3f}")

st.markdown(
    """
<div class="fuchsia-box">
    <span class="mini-title">Interpretation</span>
    <p>
        The AUC is high, which indicates a strong overall separation between genuine and impostor comparisons.
        The EER remains relatively low, confirming that the system preserves good verification performance even when twin-related comparisons are explicitly included.
        In other words, a measurable twin effect may exist, but the global discrimination capability is still robust.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

count_col, stat_col = st.columns([0.95, 1.45])
with count_col:
    st.subheader("Pair counts by relation")
    st.markdown(render_table_html(format_relation_df(DISPLAY_COUNTS)), unsafe_allow_html=True)

with stat_col:
    st.subheader("Hamming-distance descriptive statistics")
    stats_display = format_relation_df(DISPLAY_HD_STATS).copy()
    for col in ["mean", "median", "std", "min", "max"]:
        stats_display[col] = stats_display[col].map(lambda x: f"{x:.4f}")
    st.markdown(render_table_html(stats_display), unsafe_allow_html=True)

# ============================================================
# SECTION 2
# ============================================================
st.header("2. Empirical Verification Results")
st.markdown(
    """
<div class="section-text">
The system compares iris-code pairs through <strong>Masked Hamming Distance (HD)</strong>, where lower values indicate stronger similarity.
The most important stress class is <strong>twin same-eye</strong>, because if that class remains inside the impostor region, the system is not being fooled by genetic similarity.
</div>
""",
    unsafe_allow_html=True,
)

if HAS_REAL_DATA:
    n_tot = len(DF_REAL)
    n_gen = len(DF_REAL[DF_REAL["relation"] == "genuine"])
    n_twin = len(DF_REAL[DF_REAL["relation"] == "twin_same_eye"])
    n_unrel = len(DF_REAL[DF_REAL["relation"] == "unrelated"])

    empirical_cols = st.columns(4)
    empirical_cols[0].metric("Loaded comparisons", f"{n_tot:,}")
    empirical_cols[1].metric("Loaded genuine", f"{n_gen:,}")
    empirical_cols[2].metric("Loaded twin same-eye", f"{n_twin:,}")
    empirical_cols[3].metric("Loaded unrelated", f"{n_unrel:,}")
else:
    st.warning("The detailed CSV for empirical plots was not found, so only the reported summary metrics are displayed.")

# ============================================================
# SECTION 3
# ============================================================
st.header("3. Evaluation Graphs")
st.markdown(
    """
<div class="section-text">
These are the four core evaluation plots: <strong>ROC</strong>, <strong>DET</strong>, <strong>scatter plot</strong>, and <strong>boxplot</strong>.
Together they show the verification trade-off and the score distribution of each relation class.
</div>
""",
    unsafe_allow_html=True,
)

if HAS_REAL_DATA:
    STATIC_FIGURES = get_static_figures(DF_REAL)

    top_left, top_right = st.columns(2)
    with top_left:
        st.subheader("ROC curve")
        st.pyplot(STATIC_FIGURES["roc"], use_container_width=True)
    with top_right:
        st.subheader("DET curve")
        st.pyplot(STATIC_FIGURES["det"], use_container_width=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.subheader("Scatter plot")
        st.pyplot(STATIC_FIGURES["scatter"], use_container_width=True)
    with bottom_right:
        st.subheader("Boxplot")
        st.pyplot(STATIC_FIGURES["boxplot"], use_container_width=True)

    st.info(
        "**How to read them:** the ROC and DET curves summarize the global verification trade-off, while the scatter plot and boxplot show how the different relation classes are positioned. "
        "The most important observation is that genuine scores stay clearly lower, whereas twin classes remain within the impostor spectrum even if they shift slightly left compared with unrelated pairs."
    )
else:
    st.warning("Graph rendering requires `demo_orale_strict_sample_v3/tables/sample_matching_results.csv` with at least the columns `relation` and `hd`.")

# ============================================================
# SECTION 4
# ============================================================
st.header("4. Pipeline Artifacts")
st.markdown(
    """
<div class="section-text">
The final Hamming-distance values are only meaningful if the iris region has been segmented and encoded correctly.
For this reason, the demo also includes the main intermediate artifacts of the pipeline.
</div>
""",
    unsafe_allow_html=True,
)

if PIPELINE_IMAGES:
    artifact_captions = {
        "Raw selected images": "Selected raw samples from the strict subset.",
        "Segmentation overlays": "Strict Daugman-style segmentation with pupil and iris boundaries.",
        "Normalized strips": "Rubber Sheet normalized iris strips.",
        "Conservative masks": "Final masks used to keep only reliable iris texture.",
        "Masked normalized strips": "Normalized strips after conservative masking.",
        "Gabor code previews": "Binary Gabor-style iris code previews derived from the normalized strips.",
    }
    for artifact_name, artifact_path in PIPELINE_IMAGES.items():
        with st.expander(f"View {artifact_name}", expanded=False):
            st.image(
                str(artifact_path),
                caption=artifact_captions.get(artifact_name, artifact_name),
                use_container_width=True,
            )
else:
    st.warning("Pipeline artifact images were not found. Run the generation script to display segmentation and encoding stages.")

st.markdown(
    """
<div class="fuchsia-box">
    <span class="mini-title">Pipeline reading</span>
    <p>
        The strict version of the pipeline deliberately keeps only reliable texture.
        This is important because any final decision becomes easier to justify: the matcher is operating mainly on valid masked bits rather than on eyelids, reflections, sclera contamination, or unstable boundaries.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SECTION 5
# ============================================================
st.header("5. Interactive Threshold and Noise Simulation")
st.markdown(
    """
<div class="section-text">
This block uses the real Hamming-distance scores loaded from the demo CSV. At noise = 0, FAR, FRR and EER are computed directly from the actual genuine and impostor/twin comparisons. The noise slider does not replace the data with a Gaussian model: it applies a controlled stress transformation to the real score arrays to show how the decision boundary becomes more fragile when acquisition or segmentation quality degrades.
</div>
""",
    unsafe_allow_html=True,
)

def live_decision_stress_test():
    live_arrays = get_real_live_arrays(DF_REAL if HAS_REAL_DATA else None)

    if live_arrays is None:
        st.warning("Real live simulation requires `demo_orale_strict_sample_v3/tables/sample_matching_results.csv` with `relation` and `hd` columns.")
        return

    default_threshold = float(DISPLAY_GLOBAL.get("EER_threshold", LATEST_GLOBAL["EER_threshold"]))

    slider_col1, slider_col2 = st.columns(2)
    with slider_col1:
        threshold = st.slider(
            "Separation threshold / decision boundary",
            min_value=0.20,
            max_value=0.55,
            value=min(max(default_threshold, 0.20), 0.55),
            step=0.001,
            format="%.3f",
            key="threshold_live_real",
        )
    with slider_col2:
        noise = st.slider(
            "Environmental / segmentation noise factor",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key="noise_live_real",
        )

    genuine, impostor, genuine_jitter, impostor_jitter = live_arrays
    genuine_live, impostor_live = stress_real_scores(genuine, impostor, genuine_jitter, impostor_jitter, noise)
    far, frr = compute_far_frr_from_arrays(threshold, genuine_live, impostor_live)
    eer, eer_threshold = compute_eer_from_score_arrays(genuine_live, impostor_live)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("False Acceptance Rate (FAR)", f"{far:.2f}%", delta="security risk" if far > 1 else "low")
    metric_col2.metric("False Rejection Rate (FRR)", f"{frr:.2f}%", delta="usability issue" if frr > 1 else "low")
    metric_col3.metric("Empirical EER", f"{eer*100:.2f}%", delta=f"HD {eer_threshold:.3f}")
    if abs(threshold - eer_threshold) <= 0.002:
        metric_col4.markdown('<div class="status-good">AT EER POINT</div>', unsafe_allow_html=True)
    else:
        metric_col4.markdown('<div class="status-risk">CUSTOM THRESHOLD</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="section-text"><b>Reading:</b> FAR and FRR are measured at the current threshold. The dotted fuchsia line marks the empirical EER threshold for the currently stressed real scores. At noise = 0, this reproduces the demo scores directly: {len(genuine)} genuine comparisons and {len(impostor)} impostor/twin comparisons.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="live-panel">', unsafe_allow_html=True)
    fig, _, _, _, _, _, _ = plot_live_real_distribution_plotly(threshold, noise, live_arrays)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "staticPlot": False,
            "responsive": True,
            "scrollZoom": False,
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

if hasattr(st, "fragment"):
    live_decision_stress_test = st.fragment(live_decision_stress_test)

live_decision_stress_test()

# ============================================================
# SECTION 6
# ============================================================
st.header("6. Final Interpretation")
st.markdown(
    """
The final message is clear and balanced. The experiment does reveal a **twin effect**: twin scores can be slightly lower than completely unrelated impostor scores, so shared genetics are not entirely invisible in the score distribution.
However, that shift is still not strong enough to make twins behave like genuine matches.

This is the central conclusion: the **genuine distribution remains distinct**, because it compares two acquisitions of the same physical iris, whereas twins still correspond to two different irises, even when they share the same DNA.
The strict Daugman-style pipeline therefore preserves the biometric assumption of iris individuality.

A second important point concerns interpretation of borderline cases. When errors appear, they are explained more convincingly by **image quality and preprocessing effects** than by genetics alone.
Imperfect segmentation, eyelash occlusions, reflections, and reduced valid-bit ratio can enlarge the overlap region and slightly degrade the final decision.
So the practical vulnerability is not simply “being a twin”, but the interaction between biological similarity and non-ideal acquisition conditions.

In summary, the project confirms that twins are a meaningful stress case and that a mild score shift exists, but it also shows that iris recognition remains substantially robust when segmentation and masking are applied carefully.
"""
)

st.info(
    "**Takeaway:** identical DNA does not imply identical iris codes. Twins may slightly shift the impostor distribution, but the system still separates identity from genetic resemblance, especially when valid iris texture is extracted cleanly."
)
