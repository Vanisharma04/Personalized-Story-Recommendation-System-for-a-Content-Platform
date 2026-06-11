"""StoryRec — an editorial, premium dashboard for the recommendation engine.

A "fine library" aesthetic: ivory paper, antique gold & burgundy, classic
serif typography, gilded book-plate recommendation cards, and a custom light
Plotly theme. Run from the project root:

    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.recommender import Recommender

st.set_page_config(
    page_title="StoryRec · A Curated Reading Companion",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ palette
INK = "#2a2118"
INK_SOFT = "#5c4f3d"
INK_FAINT = "#8a7a63"
PAPER = "#f7f2e9"
GOLD = "#b08d4e"
GOLD_LT = "#c9a95f"
GOLD_DK = "#9a7635"
BURGUNDY = "#7a2230"

# muted antique jewel tones — readable on cream, never neon
GENRE_COLORS = {
    "Fantasy": "#6b4e8c",          # amethyst
    "Romance": "#a8536b",          # antique rose
    "Mystery": "#3d5a6c",          # slate blue
    "Science Fiction": "#2f6b68",  # antique teal
    "Horror": "#7a2e2e",           # oxblood
    "Historical Fiction": "#8a6d3b",  # bronze
    "Thriller": "#4a4e57",         # charcoal
    "Adventure": "#5a7344",        # olive
    "Drama": "#b5683f",            # terracotta
    "Mythology": "#473f73",        # indigo
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;0,800;1,500&family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

:root {
  --ink:#2a2118; --ink-soft:#5c4f3d; --ink-faint:#8a7a63;
  --gold:#b08d4e; --gold-lt:#c9a95f; --gold-dk:#9a7635; --burgundy:#7a2230;
  --paper:#f7f2e9; --card:#fbf8f1; --line:#e4d9c2;
}

/* ---------- page canvas: subtle aged-paper wash ---------- */
.stApp {
  background:
    radial-gradient(circle at 18% 8%, rgba(176,141,78,0.06), transparent 42%),
    radial-gradient(circle at 85% 92%, rgba(122,34,48,0.05), transparent 45%),
    linear-gradient(180deg, #f9f4ec 0%, #f4eddf 100%);
}
.block-container { max-width: 1180px; padding-top: 3.8rem; padding-bottom: 4rem; }

/* ---------- typography ---------- */
html, body, [class*="css"] { font-family:'EB Garamond', Georgia, serif; color: var(--ink); }
h1, h2, h3, h4 { font-family:'Playfair Display', Georgia, serif !important; color: var(--ink) !important; letter-spacing:.2px; }
p, li, span, label, .stMarkdown { font-family:'EB Garamond', Georgia, serif; font-size:1.04rem; }
a { color: var(--burgundy); }

/* ---------- gilded hero ---------- */
.hero { text-align:center; padding: 18px 10px 6px; animation: rise .9s ease both; }
.hero-eyebrow {
  font-family:'EB Garamond', serif; text-transform:uppercase; letter-spacing:.42em;
  font-size:.72rem; color: var(--gold-dk); font-weight:500;
  display:flex; align-items:center; justify-content:center; gap:16px; margin-bottom:10px;
}
.hero-eyebrow::before, .hero-eyebrow::after {
  content:""; height:1px; width:54px;
  background:linear-gradient(90deg, transparent, var(--gold)); }
.hero-eyebrow::after { background:linear-gradient(90deg, var(--gold), transparent); }
.hero-title {
  font-family:'Playfair Display', serif; font-weight:800;
  font-size: 4.4rem; line-height:1.02; margin: 4px 0 2px;
  color: var(--ink);
}
.hero-title .amp { color: var(--gold-dk); font-style:italic; font-weight:500; }
.hero-sub {
  font-family:'Cormorant Garamond', serif; font-style:italic;
  font-size: 1.5rem; color: var(--ink-soft); margin: 6px auto 4px; max-width: 720px;
}
.fleuron { color: var(--gold); font-size:1.3rem; letter-spacing:.6em; margin:14px 0 6px;
  animation: shimmer 4s ease-in-out infinite; }
.gold-rule { height:2px; width:200px; margin:6px auto 8px;
  background:linear-gradient(90deg, transparent, var(--gold), transparent); }

/* ---------- section headers ---------- */
.sec { display:flex; align-items:center; gap:16px; margin: 30px 0 14px; }
.sec-fleuron { color: var(--gold); font-size:1.15rem; }
.sec-title { font-family:'Playfair Display', serif; font-weight:700; font-size:1.7rem; color:var(--ink); white-space:nowrap; }
.sec-rule { flex:1; height:1px; background:linear-gradient(90deg, var(--line), transparent); }
.sec-sub { font-family:'Cormorant Garamond', serif; font-style:italic; font-size:1.18rem;
  color: var(--ink-faint); margin:-6px 0 12px 30px; }

/* ---------- page header (sub-pages) ---------- */
.page-eyebrow { text-transform:uppercase; letter-spacing:.36em; font-size:.72rem;
  color: var(--gold-dk); margin-bottom:2px; padding-top:6px; line-height:1.6; }
.page-title { font-family:'Playfair Display', serif; font-weight:800; font-size:2.9rem;
  margin:0 0 2px; color:var(--ink); animation: rise .7s ease both; }
.page-sub { font-family:'Cormorant Garamond', serif; font-style:italic; font-size:1.3rem;
  color: var(--ink-soft); margin-bottom: 10px; }

/* ---------- metric cards ---------- */
div[data-testid="stMetric"] {
  background: linear-gradient(160deg, #fdfbf6 0%, #f6efe0 100%);
  border: 1px solid var(--line); border-top: 2px solid var(--gold);
  border-radius: 4px; padding: 16px 20px 14px;
  box-shadow: 0 10px 26px -20px rgba(42,33,24,.6);
  transition: transform .35s ease, box-shadow .35s ease; animation: rise .8s ease both;
}
div[data-testid="stMetric"]:hover { transform: translateY(-3px);
  box-shadow: 0 18px 34px -22px rgba(42,33,24,.7); }
div[data-testid="stMetricLabel"] p {
  text-transform:uppercase; letter-spacing:.18em; font-size:.7rem !important;
  color: var(--gold-dk) !important; font-weight:500; }
div[data-testid="stMetricValue"] {
  font-family:'Playfair Display', serif !important; font-weight:700;
  font-size: 2.2rem !important; color: var(--ink) !important; }

/* ---------- gilded book-plate cards (recommendations) ---------- */
.book-card {
  display:flex; gap:22px; align-items:flex-start;
  background: linear-gradient(135deg, #fdfbf6 0%, #f7f0e2 100%);
  border: 1px solid var(--line); border-left: 3px solid var(--gold);
  border-radius: 5px; padding: 20px 26px; margin-bottom: 15px;
  box-shadow: 0 12px 30px -24px rgba(42,33,24,.75);
  transition: transform .4s cubic-bezier(.2,.7,.2,1), box-shadow .4s ease, border-color .4s ease;
  animation: rise .7s ease both;
}
.book-card:hover { transform: translateY(-4px) translateX(2px);
  box-shadow: 0 22px 44px -26px rgba(42,33,24,.8); border-left-color: var(--burgundy); }
.book-numeral {
  font-family:'Playfair Display', serif; font-weight:800; font-size: 2.9rem;
  line-height:1; min-width: 64px; text-align:center;
  background: linear-gradient(160deg, var(--gold-lt), var(--gold-dk));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.book-title { font-family:'Playfair Display', serif; font-weight:700; font-size:1.32rem;
  color: var(--ink); line-height:1.2; }
.book-byline { font-family:'Cormorant Garamond', serif; font-style:italic;
  font-size:1.12rem; color: var(--ink-soft); margin:1px 0 9px; }
.book-pills { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-bottom:9px; }
.book-desc { font-size:1.0rem; color: var(--ink-soft); line-height:1.55; }
.stat { font-family:'EB Garamond', serif; text-transform:uppercase; letter-spacing:.13em;
  font-size:.7rem; color: var(--ink-faint); }
.stat b { color: var(--gold-dk); font-weight:600; letter-spacing:.05em; }
.genre-pill { display:inline-block; padding:2px 13px; border-radius:999px;
  font-size:.72rem; font-weight:500; letter-spacing:.06em; }

/* ---------- info tiles (how it works) ---------- */
.tile { background: linear-gradient(150deg,#fdfbf6,#f5eee0); border:1px solid var(--line);
  border-radius:5px; padding:16px 20px; height:100%;
  box-shadow:0 10px 26px -22px rgba(42,33,24,.6); transition:transform .35s ease;
  animation: rise .8s ease both; }
.tile:hover { transform: translateY(-3px); }
.tile-k { text-transform:uppercase; letter-spacing:.2em; font-size:.66rem; color:var(--gold-dk); }
.tile-t { font-family:'Playfair Display',serif; font-weight:700; font-size:1.12rem; margin:3px 0 4px; }
.tile-d { font-size:.96rem; color:var(--ink-soft); line-height:1.5; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family:'Cormorant Garamond', serif; font-size:1.2rem;
  font-weight:600; color: var(--ink-faint); padding-bottom:8px; }
.stTabs [aria-selected="true"] { color: var(--burgundy) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--gold) !important; height:2px; }

/* ---------- sidebar: candle-lit espresso ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(185deg, #241c14 0%, #2c2218 60%, #211a12 100%);
  border-right: 1px solid #3a2e1f;
}
section[data-testid="stSidebar"] * { color: #e9dcc2 !important; }
.brand { text-align:center; padding: 6px 0 2px; }
.brand-mark { font-family:'Playfair Display', serif; font-weight:800; font-size:1.9rem;
  color:#f3e7cc !important; letter-spacing:.5px; }
.brand-mark .dot { color: var(--gold-lt) !important; }
.brand-tag { font-family:'Cormorant Garamond', serif; font-style:italic; font-size:1.02rem;
  color:#c7b691 !important; margin-top:-2px; }
.brand-rule { height:1px; margin:14px 4px; background:linear-gradient(90deg,transparent,#7a6442,transparent); }
.brand-stat { display:flex; justify-content:space-between; padding:4px 2px; font-size:.92rem; }
.brand-stat .v { color:#f0e2c4 !important; font-weight:600; }
.brand-stat .k { color:#b29d76 !important; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; }
/* sidebar nav links */
section[data-testid="stSidebarNav"] a { border-radius:4px; transition: background .25s ease; }
section[data-testid="stSidebarNav"] a:hover { background: rgba(176,141,78,.12); }
section[data-testid="stSidebarNav"] span { font-family:'Cormorant Garamond', serif !important;
  font-size:1.1rem !important; }

/* ---------- misc ---------- */
hr { border-color: var(--line); }
.stAlert { border-radius:5px; }
[data-testid="stCaptionContainer"] { font-style:italic; color: var(--ink-faint) !important; }

/* ---------- animations ---------- */
@keyframes rise { from { opacity:0; transform: translateY(16px); } to { opacity:1; transform:none; } }
@keyframes shimmer { 0%,100% { opacity:.55; } 50% { opacity:1; } }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Lighting the lamps & opening the archive …")
def get_recommender() -> Recommender:
    return Recommender()


@st.cache_data
def load_metrics():
    path = config.REPORTS_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_json_report(name: str):
    path = config.REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_tables():
    stories = pd.read_csv(config.RAW_DIR / "stories.csv")
    users = pd.read_csv(config.RAW_DIR / "users.csv")
    interactions = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
    return stories, users, interactions


# ------------------------------------------------------------------ ui helpers
def genre_pill(genre: str) -> str:
    color = GENRE_COLORS.get(genre, INK_FAINT)
    return (f'<span class="genre-pill" style="background:{_hex_to_rgba(color, .12)};'
            f'color:{color};border:1px solid {_hex_to_rgba(color, .4)}">{genre}</span>')


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="sec"><span class="sec-fleuron">❧</span>'
        f'<span class="sec-title">{title}</span><span class="sec-rule"></span></div>'
        + (f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-eyebrow">{eyebrow}</div>'
        f'<div class="page-title">{title}</div>'
        f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def plotly_classic(fig: go.Figure, height: int = 380) -> go.Figure:
    # Only style the title when the chart actually has one — setting title_font
    # on a title-less figure makes Plotly render a literal "undefined".
    has_title = bool(fig.layout.title.text)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_hex_to_rgba(PAPER, 0.55),
        height=height,
        margin=dict(l=12, r=12, t=54 if has_title else 18, b=12),
        font=dict(family="EB Garamond, Georgia, serif", color=INK_SOFT, size=13),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        colorway=[GOLD, BURGUNDY, "#3d5a6c", "#5a7344", "#6b4e8c", "#b5683f"],
    )
    if has_title:
        fig.update_layout(title_font=dict(
            family="Cormorant Garamond, serif", size=20, color=INK))
    fig.update_xaxes(gridcolor=_hex_to_rgba(INK_FAINT, 0.16),
                     linecolor=_hex_to_rgba(INK_FAINT, 0.35), zeroline=False,
                     title_font=dict(size=12, color=INK_FAINT))
    fig.update_yaxes(gridcolor=_hex_to_rgba(INK_FAINT, 0.16),
                     linecolor=_hex_to_rgba(INK_FAINT, 0.35), zeroline=False,
                     title_font=dict(size=12, color=INK_FAINT))
    return fig


# ================================================================== pages
def page_home():
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Personalized Story Discovery</div>
      <div class="hero-title">Story<span class="amp">Rec</span></div>
      <div class="fleuron">❧&nbsp;&nbsp;✦&nbsp;&nbsp;❧</div>
      <div class="hero-sub">A curated reading companion — where transformer-read prose
      meets a two-stage engine that learns each reader's quiet preferences.</div>
      <div class="gold-rule"></div>
    </div>
    """, unsafe_allow_html=True)

    stories, users, interactions = load_tables()
    metrics = load_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readers", f"{users.shape[0]:,}")
    c2.metric("Stories", f"{stories.shape[0]:,}")
    c3.metric("Interactions", f"{len(interactions):,}")
    c4.metric("NDCG @ 10", f"{metrics['ndcg_at_10']:.3f}" if metrics else "—")

    section_header("The Craft", "Six movements, from raw prose to a personal shelf")
    tiles = [
        ("I · Understanding", "Sentence Transformers",
         "all-MiniLM-L6-v2 reads each story's title, genre & description into a 384-dimension meaning-vector."),
        ("II · The Reader", "Engagement-weighted profiles",
         "A reader becomes the weighted average of what they finished, loved & read recently (60-day half-life)."),
        ("III · Retrieval", "Two-Tower network",
         f"User & story towers meet in a shared {config.TOWER_DIM}-d space, trained with in-batch sampled softmax."),
        ("IV · The Shelf", "FAISS candidates",
         f"Exact cosine search gathers the Top-{config.TOP_K_CANDIDATES} stories worth a reader's evening."),
        ("V · Judgement", "LightGBM LambdaRank",
         "Twelve reader, story & affinity signals re-order the shelf, optimising NDCG directly."),
        ("VI · The Proof", "Honest evaluation",
         "Recall@50, NDCG@10, MAP & MRR, measured on a strict temporal hold-out."),
    ]
    cols = st.columns(3)
    for i, (k, t, d) in enumerate(tiles):
        with cols[i % 3]:
            st.markdown(f'<div class="tile"><div class="tile-k">{k}</div>'
                        f'<div class="tile-t">{t}</div>'
                        f'<div class="tile-d">{d}</div></div>', unsafe_allow_html=True)
        if i % 3 == 2 and i != len(tiles) - 1:
            cols = st.columns(3)

    section_header("From Many, a Few", "The funnel that turns a library into a recommendation")
    col1, col2 = st.columns([2, 3])
    with col1:
        fig = go.Figure(go.Funnel(
            y=["Whole library", "FAISS candidates", "Final shelf"],
            x=[len(stories), config.TOP_K_CANDIDATES, config.TOP_N_RECOMMENDATIONS],
            textinfo="value", textfont=dict(family="Playfair Display, serif", size=15),
            marker=dict(color=[GOLD, BURGUNDY, "#5a7344"],
                        line=dict(color=PAPER, width=2)),
            connector=dict(line=dict(color=_hex_to_rgba(INK_FAINT, .4))),
        ))
        fig.update_layout(title="Library → shelf, in three steps")
        st.plotly_chart(plotly_classic(fig, 330), width='stretch')
    with col2:
        genre_counts = stories["genre"].value_counts().reset_index()
        fig = px.pie(genre_counts, names="genre", values="count", hole=0.62,
                     color="genre", color_discrete_map=GENRE_COLORS)
        fig.update_traces(marker=dict(line=dict(color=PAPER, width=2)),
                          textfont=dict(family="EB Garamond, serif", size=12))
        fig.update_layout(title="A balanced catalogue, by genre")
        st.plotly_chart(plotly_classic(fig, 330), width='stretch')


def page_recommend():
    page_header("The Reading Room", "A Shelf, Curated",
                "Choose a reader and the engine composes their personal Top-N — live.")
    rec = get_recommender()
    stories, users, _ = load_tables()

    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.selectbox(
            "Select a reader", users["user_id"].tolist(), index=42,
            help="Recommendations are generated live: FAISS retrieval + LightGBM ranking.")
    with col2:
        n = st.slider("Stories to recommend", 5, 20, config.TOP_N_RECOMMENDATIONS)

    user_row = users.set_index("user_id").loc[user_id]
    history = rec.user_history(user_id)

    c1, c2, c3 = st.columns(3)
    c1.metric("Preferred genre", user_row["preferred_genre"])
    c2.metric("Age", int(user_row["age"]))
    c3.metric("Stories read", len(rec.seen.get(user_id, set())))

    tab_recs, tab_history = st.tabs(["The Recommendation", "Reading History"])

    with tab_recs:
        with st.spinner("Retrieving & ranking …"):
            recs = rec.recommend(user_id, n=n)
        section_header("Chosen for You",
                       f"{len(recs)} stories, ordered by the engine's conviction")
        for _, r in recs.iterrows():
            st.markdown(f"""
            <div class="book-card">
              <div class="book-numeral">{r['rank']:02d}</div>
              <div>
                <div class="book-title">{r['title']}</div>
                <div class="book-byline">by {r['author']}</div>
                <div class="book-pills">{genre_pill(r['genre'])}
                  <span class="stat">Similarity&nbsp;·&nbsp;<b>{r['similarity']:.3f}</b></span>
                  <span class="stat">Rank score&nbsp;·&nbsp;<b>{r['rank_score']:.3f}</b></span>
                </div>
                <div class="book-desc">{r['description']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        section_header("Why These Stories?",
                       "Retrieval affinity behind each chosen title")
        fig = px.bar(recs.sort_values("rank", ascending=False),
                     x="similarity", y="title", orientation="h",
                     color="genre", color_discrete_map=GENRE_COLORS)
        fig.update_layout(yaxis_title="", xaxis_title="cosine similarity",
                          showlegend=False)
        st.plotly_chart(plotly_classic(fig, 60 + 34 * len(recs)), width='stretch')

    with tab_history:
        if history.empty:
            st.info("No reading recorded for this reader in the training window.")
        else:
            section_header("Lately Read", "The reading that shaped this reader's profile")
            show = history[["timestamp", "title", "genre", "author",
                            "completion_rate", "reading_time", "likes"]]
            st.dataframe(show, width='stretch', hide_index=True)
            fig = px.pie(history, names="genre", hole=0.62, color="genre",
                         color_discrete_map=GENRE_COLORS)
            fig.update_traces(marker=dict(line=dict(color=PAPER, width=2)))
            fig.update_layout(title="Their recent reading mix")
            st.plotly_chart(plotly_classic(fig, 330), width='stretch')


def page_analytics():
    page_header("The Ledger", "Measures & Proof",
                "Offline evaluation on a strict temporal hold-out — the future, predicted from the past.")
    metrics = load_metrics()
    if metrics is None:
        st.warning("No metrics found — run `python pipeline.py` first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall @ 50", f"{metrics['recall_at_50']:.3f}",
              help="Share of held-out reads captured in the Top-50 FAISS candidates")
    c2.metric("NDCG @ 10", f"{metrics['ndcg_at_10']:.3f}",
              help="Ranking quality of the final Top-10 list")
    c3.metric("MAP", f"{metrics['map']:.3f}",
              help="Mean Average Precision over ranked candidates")
    c4.metric("MRR", f"{metrics['mrr']:.3f}",
              help="How early the first relevant story appears")
    st.caption(f"Evaluated on {metrics['n_users_evaluated']:,} readers · "
               f"Top-{metrics['top_k_candidates']} candidate retrieval")

    section_header("The Ranker's Worth", "What the second stage adds over raw retrieval")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=["Retrieval order", "After LightGBM ranking"],
            y=[metrics["ndcg_at_10_retrieval_only"], metrics["ndcg_at_10"]],
            marker=dict(color=["#c9bfa8", GOLD], line=dict(color=PAPER, width=1.5)),
            text=[f"{metrics['ndcg_at_10_retrieval_only']:.3f}",
                  f"{metrics['ndcg_at_10']:.3f}"],
            textposition="outside",
            textfont=dict(family="Playfair Display, serif", size=15, color=INK),
        ))
        fig.update_layout(title="NDCG@10 — lift from the ranking stage",
                          yaxis_range=[0, max(metrics["ndcg_at_10"] * 1.35, 0.1)])
        st.plotly_chart(plotly_classic(fig), width='stretch')
    with col2:
        genre_ndcg = load_json_report("genre_ndcg.json") or {}
        gdf = pd.DataFrame(genre_ndcg.items(), columns=["genre", "ndcg"]).sort_values("ndcg")
        fig = px.bar(gdf, x="ndcg", y="genre", orientation="h", color="genre",
                     color_discrete_map=GENRE_COLORS)
        fig.update_layout(showlegend=False, xaxis_title="NDCG@10", yaxis_title="",
                          title="Ranking quality by preferred genre")
        st.plotly_chart(plotly_classic(fig), width='stretch')

    stories, users, interactions = load_tables()

    section_header("The Readers' Pulse", "How taste and attention move across the catalogue")
    col1, col2 = st.columns(2)
    with col1:
        eng = interactions.merge(stories[["story_id", "genre"]], on="story_id")
        agg = eng.groupby("genre").agg(
            reads=("story_id", "count"),
            avg_completion=("completion_rate", "mean"),
            like_rate=("likes", "mean")).reset_index()
        fig = px.scatter(agg, x="avg_completion", y="like_rate", size="reads",
                         color="genre", color_discrete_map=GENRE_COLORS,
                         size_max=48)
        fig.update_traces(marker=dict(line=dict(color=PAPER, width=1.5)))
        fig.update_layout(xaxis_title="avg completion rate", yaxis_title="like rate",
                          showlegend=False, title="The genre engagement landscape")
        st.plotly_chart(plotly_classic(fig), width='stretch')
    with col2:
        ts = interactions.copy()
        ts["week"] = pd.to_datetime(ts["timestamp"]).dt.to_period("W").dt.start_time
        weekly = ts.groupby("week").size().rename("reads").reset_index()
        fig = px.area(weekly, x="week", y="reads")
        fig.update_traces(line=dict(color=BURGUNDY, width=2),
                          fillcolor=_hex_to_rgba(BURGUNDY, 0.12))
        fig.update_layout(title="Reading activity over time")
        st.plotly_chart(plotly_classic(fig), width='stretch')

    importance = load_json_report("feature_importance.json")
    if importance:
        section_header("What Sways the Verdict", "LightGBM feature importance, by split count")
        idf = pd.DataFrame(importance.items(),
                           columns=["feature", "importance"]).sort_values("importance")
        fig = px.bar(idf, x="importance", y="feature", orientation="h",
                     color_discrete_sequence=["#8a6d3b"])
        fig.update_traces(marker=dict(line=dict(color=PAPER, width=1)))
        fig.update_layout(xaxis_title="splits", yaxis_title="")
        st.plotly_chart(plotly_classic(fig, 420), width='stretch')

    history = load_json_report("two_tower_history.json")
    if history:
        section_header("The Towers, Learning", "Sampled-softmax loss across training epochs")
        loss = history["loss_history"]
        fig = px.line(x=list(range(1, len(loss) + 1)), y=loss, markers=True)
        fig.update_traces(line=dict(color=BURGUNDY, width=2.5),
                          marker=dict(color=GOLD, size=8,
                                      line=dict(color=PAPER, width=1)))
        fig.update_layout(xaxis_title="epoch", yaxis_title="loss")
        st.plotly_chart(plotly_classic(fig, 320), width='stretch')


# ================================================================== nav
pages = st.navigation([
    st.Page(page_home, title="The Atelier", icon=":material/auto_stories:"),
    st.Page(page_recommend, title="Reading Room", icon=":material/menu_book:"),
    st.Page(page_analytics, title="The Ledger", icon=":material/insights:"),
])
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">Story<span class="dot">Rec</span></div>'
        '<div class="brand-tag">a curated reading companion</div></div>'
        '<div class="brand-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="brand-stat"><span class="k">Catalogue</span>'
        f'<span class="v">{config.N_STORIES:,} stories</span></div>'
        f'<div class="brand-stat"><span class="k">Readers</span>'
        f'<span class="v">{config.N_USERS:,}</span></div>'
        f'<div class="brand-stat"><span class="k">Retrieve</span>'
        f'<span class="v">Top-{config.TOP_K_CANDIDATES}</span></div>'
        f'<div class="brand-stat"><span class="k">Rank to</span>'
        f'<span class="v">Top-{config.TOP_N_RECOMMENDATIONS}</span></div>'
        '<div class="brand-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Cormorant Garamond,serif;font-style:italic;'
        'font-size:1.0rem;color:#bda981;text-align:center;line-height:1.5;">'
        'Sentence Transformers · Two-Tower<br>FAISS · LightGBM LambdaRank</div>',
        unsafe_allow_html=True)
pages.run()
