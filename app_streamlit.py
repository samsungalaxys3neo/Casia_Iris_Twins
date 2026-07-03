import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.stats import norm
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Iris Twins Demo",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CONSTANTS
# ============================================================
PAPER_TITLE = "The Illusion of Uniqueness: Iris Recognition in Twin Subjects"
AUTHORS = "Sarah Oualli, Michele Pogu, Gaia Diodati"
FRAMEWORK = "Daugman-style strict pipeline · Segmentation · Rubber Sheet Normalization · Gabor-style Encoding"
RUN_COMMAND = "python3 scripts/demo_orale_strict_sample.py --seed 1234 --families 20 --images-per-iris 5"

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
        {"relation": "unrelated", "count": 7294},
        {"relation": "same_subject_different_eye", "count": 230},
        {"relation": "twin_cross_eye", "count": 210},
        {"relation": "twin_same_eye", "count": 202},
        {"relation": "genuine", "count": 192},
    ]
)

LATEST_HD_STATS = pd.DataFrame(
    [
        {
            "relation": "genuine",
            "count": 192,
            "mean": 0.3105201819026891,
            "median": 0.30069266214220014,
            "std": 0.06404810202911651,
            "min": 0.1676644493717664,
            "max": 0.445593455780213,
        },
        {
            "relation": "twin_same_eye",
            "count": 202,
            "mean": 0.42777398492270713,
            "median": 0.4292988055184336,
            "std": 0.026536897299942,
            "min": 0.3635978428351309,
            "max": 0.488566131025958,
        },
        {
            "relation": "twin_cross_eye",
            "count": 210,
            "mean": 0.43323789191178597,
            "median": 0.4329110191654707,
            "std": 0.028671048966570592,
            "min": 0.3668966028779113,
            "max": 0.5148257968865827,
        },
        {
            "relation": "same_subject_different_eye",
            "count": 230,
            "mean": 0.43719027938700483,
            "median": 0.43949502338197577,
            "std": 0.02567030116537567,
            "min": 0.3658262821619755,
            "max": 0.5166116694165291,
        },
        {
            "relation": "unrelated",
            "count": 345,
            "mean": 0.4419212172789568,
            "median": 0.4421918254355968,
            "std": 0.021016236511993476,
            "min": 0.3869355107482086,
            "max": 0.520506329113924,
        },
    ]
)

LATEST_GLOBAL = {
    "genuine_pairs": 192,
    "impostor_pairs": 987,
    "EER": 0.11916635005065856,
    "EER_threshold": 0.4041481734380334,
    "AUC": 0.9581275329280647,
}

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
        padding-top: 1.7rem;
        padding-bottom: 4rem;
        max-width: 1500px;
    }}

    p, li, div, span, label {{
        font-size: 1.34rem;
        line-height: 1.74;
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
        font-size: 5.35rem !important;
        line-height: 0.98 !important;
        color: var(--black) !important;
        margin-bottom: 0.85rem !important;
        max-width: 1250px;
    }}

    h2 {{
        font-size: 2.65rem !important;
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
        font-size: 1.95rem !important;
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
        padding: 2.6rem 2.65rem 2.25rem 2.65rem;
        box-shadow: 0 18px 42px rgba(17, 17, 17, 0.09);
        margin-bottom: 1.45rem;
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
        padding: 0.25rem 0.86rem 0.30rem 0.86rem;
        font-size: 1.24rem;
        letter-spacing: 0.12em;
        font-weight: 900;
        text-transform: uppercase;
        vertical-align: middle;
        margin-right: 0.60rem;
        box-shadow: 0 0 0 7px rgba(230, 0, 126, 0.08);
    }}

    .paper-title-highlight {{
        color: var(--black);
    }}

    .authors-line {{
        font-size: 1.46rem;
        font-weight: 700;
        color: var(--black);
        margin-top: 0.65rem;
    }}

    .framework-line {{
        font-size: 1.22rem;
        font-weight: 600;
        color: #454545;
        margin-top: 0.25rem;
    }}

    .lead-text {{
        font-size: 1.52rem;
        line-height: 1.72;
        color: #1f1f1f;
        max-width: 1220px;
        margin-top: 1.25rem;
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
        font-size: 1.34rem;
        color: var(--black);
        line-height: 1.68;
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
        font-size: 1.16rem !important;
        font-weight: 800 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: var(--fuchsia) !important;
        font-size: 2.65rem !important;
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
        font-size: 1.22rem !important;
        line-height: 1.62 !important;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1.5px solid #efefef;
        border-radius: 16px;
        overflow: hidden;
    }}

    div[data-testid="stImageCaption"] {{
        font-size: 1.08rem !important;
        color: #333333 !important;
        font-weight: 600;
        text-align: center;
    }}

    div.stSlider label p {{
        font-size: 1.18rem !important;
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
        font-size: 1.05rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.78rem 0.9rem;
        font-weight: 900;
    }}

    .custom-table td {{
        color: var(--black);
        font-size: 1.22rem;
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
        font-size: 1.5rem;
        font-weight: 900;
    }}

    .status-risk {{
        text-align: center;
        border: 2px solid var(--fuchsia);
        background: #fff7fb;
        color: var(--fuchsia);
        border-radius: 18px;
        padding: 1.15rem;
        font-size: 1.5rem;
        font-weight: 900;
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
        "font.size": 15,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
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
            images["Overlays"] = ref_dir / "02_segmentation_true_overlays.png"
            images["Masked Strips"] = ref_dir / "05_masked_normalized_strips.png"
            images["Gabor Codes"] = ref_dir / "06_gabor_codes.png"
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
        ax.set_title(title, fontsize=20, fontweight="bold", pad=14, color=BLACK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16, fontweight="bold", labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16, fontweight="bold", labelpad=10)
    ax.grid(True, color=LIGHT_GREY, linewidth=1.0, alpha=0.95)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#d6d6d6")
        spine.set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=12, colors=BLACK)


def add_legend(ax, loc="best"):
    legend = ax.legend(loc=loc, frameon=True, facecolor=WHITE, edgecolor="#dddddd", framealpha=1)
    if legend:
        for text in legend.get_texts():
            text.set_color(BLACK)
            text.set_fontsize(11)
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


def plot_live_distribution_plotly(threshold, new_mu_g, new_sigma_g, mu_impostor_base, new_sigma_i, far, frr):
    x = np.linspace(0.15, 0.55, 260)
    y_g = norm.pdf(x, new_mu_g, new_sigma_g)
    y_i = norm.pdf(x, mu_impostor_base, new_sigma_i)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=y_g,
        mode="lines",
        name="Genuine model",
        line=dict(color="#1f77b4", width=3),
        fill="tozeroy",
        fillcolor="rgba(31, 119, 180, 0.22)",
        hovertemplate="HD=%{x:.3f}<br>Density=%{y:.3f}<extra>Genuine</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y_i,
        mode="lines",
        name="Twin + impostor model",
        line=dict(color=FUCHSIA, width=3),
        fill="tozeroy",
        fillcolor="rgba(230, 0, 126, 0.18)",
        hovertemplate="HD=%{x:.3f}<br>Density=%{y:.3f}<extra>Impostor</extra>",
    ))

    if far > 0.01:
        x_far = np.linspace(0.15, threshold, 150)
        y_far = norm.pdf(x_far, mu_impostor_base, new_sigma_i)
        fig.add_trace(go.Scatter(
            x=x_far, y=y_far,
            mode="lines",
            name="False acceptances (FAR)",
            line=dict(color=FUCHSIA, width=0),
            fill="tozeroy",
            fillcolor="rgba(230, 0, 126, 0.28)",
            hoverinfo="skip",
        ))

    if frr > 0.01:
        x_frr = np.linspace(threshold, 0.55, 150)
        y_frr = norm.pdf(x_frr, new_mu_g, new_sigma_g)
        fig.add_trace(go.Scatter(
            x=x_frr, y=y_frr,
            mode="lines",
            name="False rejections (FRR)",
            line=dict(color="#1f77b4", width=0),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.28)",
            hoverinfo="skip",
        ))

    fig.add_vline(
        x=threshold,
        line_width=2.5,
        line_dash="dash",
        line_color=BLACK,
        annotation_text=f"Threshold = {threshold:.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(text="Interactive Distribution Model under Noise", font=dict(size=23, color=BLACK)),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        hovermode=False,
        uirevision="live",
        margin=dict(l=34, r=24, t=58, b=38),
        legend=dict(bgcolor="rgba(255,255,255,0.88)", bordercolor="#dddddd", borderwidth=1, font=dict(size=12, color=BLACK)),
        xaxis=dict(
            title="Hamming Distance (HD)",
            range=[0.15, 0.55],
            showgrid=True,
            gridcolor=LIGHT_GREY,
            zeroline=False,
            tickfont=dict(size=13, color=BLACK),
            title_font=dict(size=16, color=BLACK),
        ),
        yaxis=dict(
            title="Probability Density",
            showgrid=True,
            gridcolor=LIGHT_GREY,
            zeroline=False,
            tickfont=dict(size=13, color=BLACK),
            title_font=dict(size=16, color=BLACK),
        ),
    )
    return fig

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
<div class="hero-card">
    <div class="cover-kicker">Interactive biometric project</div>
    <h1><span class="demo-badge">DEMO</span><span class="paper-title-highlight">{PAPER_TITLE}</span></h1>
    <div class="authors-line"><b>Authors:</b> {AUTHORS}</div>
    <div class="framework-line">{FRAMEWORK}</div>
    <p class="lead-text">
        This interactive demo summarizes the biometric evidence of the paper in a clean white layout with fuchsia accents.
        The central question is whether monozygotic twins, despite sharing the same DNA, produce iris patterns similar enough to challenge a classical iris-recognition pipeline.
        The analysis is presented through strict segmentation, conservative masking, and Hamming-distance matching.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="fuchsia-box">
    <span class="mini-title">Core idea</span>
    <p>
        Iris recognition is generally considered highly reliable because the fine iris texture emerges from stochastic developmental processes.
        Twins are therefore an ideal stress case: if a measurable similarity exists, it should appear as a shift of twin impostor scores toward the genuine region.
        The real issue is not simply whether twins are a bit more similar than unrelated people, but whether that shift becomes strong enough to blur the decision boundary.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

DF_REAL = load_real_data()
PIPELINE_IMAGES = get_image_paths()
HAS_REAL_DATA = DF_REAL is not None and {"relation", "hd"}.issubset(set(DF_REAL.columns))

# ============================================================
# SECTION 1
# ============================================================
st.header("1. Latest Run Summary")
st.markdown(
    f"""
This section reports the latest demo run generated with:
`{RUN_COMMAND}`.
The values below summarize the key verification outcomes and relation-level statistics.
"""
)

metric_row_1 = st.columns(5)
metric_row_1[0].metric("Genuine pairs", f"{LATEST_GLOBAL['genuine_pairs']}")
metric_row_1[1].metric("Impostor pairs", f"{LATEST_GLOBAL['impostor_pairs']}")
metric_row_1[2].metric("EER", f"{LATEST_GLOBAL['EER']*100:.2f}%")
metric_row_1[3].metric("EER threshold", f"{LATEST_GLOBAL['EER_threshold']:.3f}")
metric_row_1[4].metric("AUC", f"{LATEST_GLOBAL['AUC']:.3f}")

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
    st.markdown(render_table_html(format_relation_df(LATEST_COUNTS)), unsafe_allow_html=True)

with stat_col:
    st.subheader("Hamming-distance descriptive statistics")
    stats_display = format_relation_df(LATEST_HD_STATS).copy()
    for col in ["mean", "median", "std", "min", "max"]:
        stats_display[col] = stats_display[col].map(lambda x: f"{x:.4f}")
    st.markdown(render_table_html(stats_display), unsafe_allow_html=True)

# ============================================================
# SECTION 2
# ============================================================
st.header("2. Empirical Verification Results")
st.markdown(
    """
The system compares iris-code pairs through **Masked Hamming Distance (HD)**, where lower values indicate stronger similarity.
The most important stress class is **twin same-eye**, because if that class remains inside the impostor region, the system is not being fooled by genetic similarity.
"""
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
st.header("3. Required Evaluation Graphs")
st.markdown(
    """
These are the four core evaluation plots: **ROC**, **DET**, **scatter plot**, and **boxplot**.
They are all rendered on a white background with black typography and fuchsia accents to keep them visually consistent and easier to read.
"""
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
The final Hamming-distance values are only meaningful if the iris region has been segmented and encoded correctly.
For this reason, the demo also includes the main intermediate artifacts of the pipeline.
"""
)

if PIPELINE_IMAGES:
    img_cols = st.columns(3)
    if "Overlays" in PIPELINE_IMAGES:
        img_cols[0].image(
            str(PIPELINE_IMAGES["Overlays"]),
            caption="Strict Daugman-style segmentation: pupil and iris boundaries after quality filtering.",
            use_container_width=True,
        )
    if "Masked Strips" in PIPELINE_IMAGES:
        img_cols[1].image(
            str(PIPELINE_IMAGES["Masked Strips"]),
            caption="Rubber Sheet normalization with conservative masking of noisy regions.",
            use_container_width=True,
        )
    if "Gabor Codes" in PIPELINE_IMAGES:
        img_cols[2].image(
            str(PIPELINE_IMAGES["Gabor Codes"]),
            caption="Binary Gabor-style iris codes derived from the normalized strips.",
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
st.header("5. Interactive Decision Stress Test")
st.markdown(
    """
This last block is a simplified live simulation of the verification process.
As noise increases, the genuine distribution becomes broader and shifts rightward, which means that genuine samples become less compact because fewer reliable bits are available.
The live graph uses Plotly and cached static sections, so slider updates are much faster.
"""
)

def live_decision_stress_test():
    mu_genuine_base = 0.31
    sigma_genuine_base = 0.035
    mu_impostor_base = 0.44
    sigma_impostor_base = 0.025

    slider_col1, slider_col2 = st.columns(2)
    with slider_col1:
        threshold = st.slider(
            "Separation threshold / decision boundary",
            min_value=0.20,
            max_value=0.50,
            value=0.360,
            step=0.005,
            format="%.3f",
            key="threshold_live",
        )
    with slider_col2:
        noise = st.slider(
            "Environmental / segmentation noise factor",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            key="noise_live",
        )

    new_sigma_g = sigma_genuine_base + (noise * 0.03)
    new_sigma_i = sigma_impostor_base + (noise * 0.02)
    new_mu_g = mu_genuine_base + (noise * 0.04)
    far, frr = calculate_metrics(threshold, new_mu_g, new_sigma_g, mu_impostor_base, new_sigma_i)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("False Acceptance Rate (FAR)", f"{far:.3f}%", delta="security risk" if far > 1 else "low")
    metric_col2.metric("False Rejection Rate (FRR)", f"{frr:.3f}%", delta="usability issue" if frr > 1 else "low")
    if far < 1.0 and frr < 1.0:
        metric_col3.markdown('<div class="status-good">OPTIMIZED REGION</div>', unsafe_allow_html=True)
    else:
        metric_col3.markdown('<div class="status-risk">OVERLAP / TRADE-OFF</div>', unsafe_allow_html=True)

    st.markdown('<div class="live-panel">', unsafe_allow_html=True)
    st.plotly_chart(
        plot_live_distribution_plotly(threshold, new_mu_g, new_sigma_g, mu_impostor_base, new_sigma_i, far, frr),
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
