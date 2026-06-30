import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path

# ─────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #0d1137 40%, #130d2e 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px;}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1f5e 0%, #2d1b69 50%, #1a0e3d 100%);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(108,99,255,0.2);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 60%, #6c63ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.55);
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(108,99,255,0.2);
    border: 1px solid rgba(108,99,255,0.5);
    color: #a78bfa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* ── Search container ── */
.search-wrapper {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

/* ── Streamlit selectbox styling ── */
/* ===========================
   Selectbox Container
=========================== */
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(26, 31, 78, 0.9) !important;
    border: 1.5px solid rgba(108,99,255,0.5) !important;
    border-radius: 14px !important;
    min-height: 54px !important;
}

/* Text typed by the user */
.stSelectbox input {
    color: black !important;
    -webkit-text-fill-color: black !important;
}

/* Placeholder text */
.stSelectbox input::placeholder {
    color: gray !important;
}

/* Selected text */
.stSelectbox div[data-baseweb="select"] * {
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* SVG arrow */
.stSelectbox svg {
    fill: white !important;
}

/* Dropdown menu */
div[role="listbox"] {
    background: #1a1f4e !important;
}

/* Dropdown options */
div[role="option"] {
    color: white !important;
    background: #1a1f4e !important;
}

/* Hover option */
div[role="option"]:hover {
    background: rgba(108,99,255,.3) !important;
}

/* Label */
.stSelectbox label {
    color: rgba(255,255,255,.75) !important;
}



/* ── Recommend button ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff 0%, #a78bfa 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 2.5rem !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 8px 25px rgba(108,99,255,0.45) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    height: 54px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(108,99,255,0.6) !important;
    background: linear-gradient(135deg, #7c73ff 0%, #b79bff 100%) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Section title ── */
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, rgba(108,99,255,0.8), transparent);
    border-radius: 2px;
    margin-bottom: 1.8rem;
}

/* ── Movie card ── */
.movie-card {
    background: linear-gradient(145deg, rgba(26,31,78,0.95), rgba(15,18,50,0.95));
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 20px;
    overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    cursor: pointer;
    position: relative;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
}
.movie-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 50px rgba(108,99,255,0.35);
    border-color: rgba(108,99,255,0.6);
}
.movie-card-rank {
    position: absolute;
    top: 10px;
    left: 10px;
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: white;
    font-size: 0.7rem;
    font-weight: 800;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
    box-shadow: 0 4px 10px rgba(108,99,255,0.5);
}
.movie-card img {
    width: 100%;
    height: 280px;
    object-fit: cover;
    display: block;
}
.movie-card-no-poster {
    width: 100%;
    height: 280px;
    background: linear-gradient(145deg, #1e2346, #252a5a);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
}
.movie-card-no-poster span {
    font-size: 3rem;
}
.movie-card-no-poster p {
    color: rgba(255,255,255,0.3);
    font-size: 0.8rem;
}
.movie-card-body {
    padding: 1rem 1.1rem 1.2rem;
}
.movie-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: white;
    margin: 0 0 0.45rem 0;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 2.6rem;
}
.movie-card-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.meta-chip {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}
.chip-year {
    background: rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.6);
}
.chip-rating {
    background: rgba(255,215,0,0.12);
    color: #ffd700;
}
.chip-genre {
    background: rgba(108,99,255,0.18);
    color: #a78bfa;
}
.movie-card-overview {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.45);
    line-height: 1.55;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-top: 0.5rem;
}

/* ── Selected movie hero ── */
.selected-hero {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(167,139,250,0.08));
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 24px;
    padding: 1.8rem 2rem;
    margin-bottom: 2.2rem;
    display: flex;
    gap: 2rem;
    align-items: flex-start;
}
.selected-poster {
    border-radius: 16px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    min-width: 150px;
}
.selected-info {
    flex: 1;
}
.selected-tag {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: #6c63ff;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.selected-title {
    font-size: 1.8rem;
    font-weight: 900;
    color: white;
    margin: 0 0 0.6rem 0;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.selected-overview {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    line-height: 1.65;
    margin-top: 0.8rem;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.stat-item {
    text-align: center;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #a78bfa;
}
.stat-label {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.4);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Popular tags ── */
.popular-chip {
    display: inline-block;
    background: rgba(26,31,78,0.8);
    border: 1px solid rgba(108,99,255,0.3);
    color: rgba(255,255,255,0.7);
    font-size: 0.8rem;
    font-weight: 500;
    padding: 6px 16px;
    border-radius: 20px;
    margin: 4px;
    cursor: pointer;
    transition: all 0.2s;
}
.popular-chip:hover {
    background: rgba(108,99,255,0.2);
    border-color: rgba(108,99,255,0.6);
    color: white;
}

/* ── Info box ── */
.info-box {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1.5rem;
    color: rgba(255,255,255,0.55);
    font-size: 0.85rem;
    line-height: 1.6;
}
.info-box strong { color: #a78bfa; }

/* ── Spinner override ── */
.stSpinner { color: #6c63ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  DATA LOADING  (cached)
# ─────────────────────────────────────────
TMDB_API_KEY = "2bbc4f508dcf31dc495ea58c2df6afaf"
TMDB_BASE    = "https://api.themoviedb.org/3"
IMG_BASE     = "https://image.tmdb.org/t/p/w500"

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner=False)
def load_data():
    with open(BASE_DIR / "movies_dict.pkl", "rb") as f:
        movies_dict = pickle.load(f)

    with open(BASE_DIR / "similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    df = pd.DataFrame(movies_dict).reset_index(drop=True)
    return df, similarity

@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id: int) -> dict:
    try:
        r = requests.get(
            f"{TMDB_BASE}/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=8,
        )
        if r.status_code == 200:
            d = r.json()
            genres = " · ".join(g["name"] for g in d.get("genres", [])[:3])
            return {
                "poster": IMG_BASE + d["poster_path"] if d.get("poster_path") else None,
                "overview": d.get("overview", ""),
                "year": (d.get("release_date") or "")[:4],
                "rating": round(d.get("vote_average", 0), 1),
                "genres": genres,
                "runtime": d.get("runtime"),
                "tagline": d.get("tagline", ""),
            }
    except Exception:
        pass
    return {"poster": None, "overview": "", "year": "", "rating": 0, "genres": "", "runtime": None, "tagline": ""}

def recommend(movie_title: str, df, similarity):
    idx = df[df["title"] == movie_title].index[0]
    distances = similarity[idx]
    top5 = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:6]
    return [(df.iloc[i].title, int(df.iloc[i].id)) for i, _ in top5]


# ─────────────────────────────────────────
#  HELPERS – HTML builders
# ─────────────────────────────────────────
def movie_card_html(title: str, details: dict, rank: int) -> str:
    poster_html = (
        f'<img src="{details["poster"]}" alt="{title}" loading="lazy"/>'
        if details["poster"]
        else '<div class="movie-card-no-poster"><span>🎬</span><p>No Poster</p></div>'
    )
    year_chip   = f'<span class="meta-chip chip-year">📅 {details["year"]}</span>' if details["year"] else ""
    rating_chip = f'<span class="meta-chip chip-rating">⭐ {details["rating"]}</span>' if details["rating"] else ""
    genre_chip  = f'<span class="meta-chip chip-genre">{details["genres"].split(" · ")[0]}</span>' if details["genres"] else ""
    overview    = details["overview"][:140] + "…" if len(details["overview"]) > 140 else details["overview"]

    return f"""
<div class="movie-card">
  <div class="movie-card-rank">#{rank}</div>
  {poster_html}
  <div class="movie-card-body">
    <p class="movie-card-title">{title}</p>
    <div class="movie-card-meta">
      {year_chip}{rating_chip}{genre_chip}
    </div>
    <p class="movie-card-overview">{overview}</p>
  </div>
</div>"""


def selected_hero_html(title: str, details: dict) -> str:
    poster_part = (
        f'<img class="selected-poster" src="{details["poster"]}" width="150" alt="{title}"/>'
        if details["poster"]
        else '<div style="min-width:150px;height:220px;background:rgba(108,99,255,0.1);border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:3rem;">🎬</div>'
    )
    rating_html  = f'<span class="meta-chip chip-rating" style="font-size:0.85rem;">⭐ {details["rating"]} / 10</span>' if details["rating"] else ""
    year_html    = f'<span class="meta-chip chip-year" style="font-size:0.85rem;">📅 {details["year"]}</span>' if details["year"] else ""
    runtime_html = f'<span class="meta-chip chip-genre" style="font-size:0.85rem;">⏱ {details["runtime"]} min</span>' if details.get("runtime") else ""
    tagline_html = f'<p style="color:#a78bfa;font-style:italic;font-size:0.9rem;margin-top:0.4rem;">"{details["tagline"]}"</p>' if details.get("tagline") else ""
    genres_html  = f'<p style="color:rgba(255,255,255,0.45);font-size:0.8rem;margin:0.5rem 0 0 0;">{details["genres"]}</p>' if details["genres"] else ""
    overview_cut = details["overview"][:300] + "…" if len(details["overview"]) > 300 else details["overview"]

    return f"""
<div class="selected-hero">
  {poster_part}
  <div class="selected-info">
    <p class="selected-tag">🎯 You searched for</p>
    <h2 class="selected-title">{title}</h2>
    {tagline_html}
    <div class="movie-card-meta" style="margin:0.5rem 0;">{rating_html}{year_html}{runtime_html}</div>
    {genres_html}
    <p class="selected-overview">{overview_cut}</p>
  </div>
</div>"""


# ─────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────
with st.spinner("Loading movie database…"):
    df, similarity = load_data()

movie_titles = df["title"].tolist()


# ─────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">🎬 AI-Powered · 4,806 Movies</div>
  <h1 class="hero-title">CineMatch</h1>
  <p class="hero-subtitle">Discover your next favorite film with machine-learning similarity search</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  SEARCH SECTION
# ─────────────────────────────────────────


col_select, col_btn = st.columns([4, 1], gap="medium")

with col_select:
    selected_movie = st.selectbox(
        "🎬  Search a movie title",
        options=["— Select a movie —"] + sorted(movie_titles),
        index=0,
        key="movie_select",
    )

with col_btn:
    st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)
    recommend_clicked = st.button("✨ Recommend", key="rec_btn", use_container_width=True)

# Popular picks
popular = ["Avatar", "The Dark Knight", "Inception", "Interstellar",
           "The Avengers", "Titanic", "Jurassic Park", "The Matrix"]
st.markdown(
    "<p style='color:rgba(255,255,255,0.35);font-size:0.78rem;margin:1rem 0 0.4rem;letter-spacing:0.5px;'>POPULAR PICKS</p>",
    unsafe_allow_html=True,
)
pop_chips_html = "".join(f'<span class="popular-chip">🍿 {m}</span>' for m in popular)
st.markdown(f"<div>{pop_chips_html}</div>", unsafe_allow_html=True)




# ─────────────────────────────────────────
#  RECOMMENDATION RESULTS
# ─────────────────────────────────────────
if recommend_clicked and selected_movie != "— Select a movie —":

    # ── Fetch selected movie details ──
    selected_id = int(df[df["title"] == selected_movie]["id"].values[0])
    with st.spinner("Fetching movie details…"):
        selected_details = fetch_movie_details(selected_id)

    # ── Hero card for selected movie ──
    st.markdown(selected_hero_html(selected_movie, selected_details), unsafe_allow_html=True)

    # ── Compute recommendations ──
    recs = recommend(selected_movie, df, similarity)

    # ── Section header ──
    st.markdown("""
    <div class="section-title">
        <span>✨</span>
        <span>Top 5 Recommendations</span>
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # ── Fetch details for all 5 movies ──
    with st.spinner("Loading recommendations…"):
        rec_details = []
        for title, mid in recs:
            rec_details.append((title, fetch_movie_details(mid)))

    # ── Render 5 cards in columns ──
    cols = st.columns(5, gap="small")
    for idx, (col, (title, details)) in enumerate(zip(cols, rec_details)):
        with col:
            st.markdown(movie_card_html(title, details, idx + 1), unsafe_allow_html=True)

    # ── Info box ──
    st.markdown(f"""
    <div class="info-box">
        <strong>How it works:</strong> CineMatch uses a <strong>cosine similarity model</strong>
        trained on {len(df):,} movies. Each film is represented as a vector of tags (plot keywords,
        genres, cast, crew, director). The 5 movies shown above have the <strong>highest similarity
        score</strong> to <em>{selected_movie}</em>.
    </div>
    """, unsafe_allow_html=True)

elif recommend_clicked and selected_movie == "— Select a movie —":
    st.warning("⚠️ Please select a movie from the dropdown first!", icon="🎬")

else:
    # ── Empty state ──
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size:5rem; margin-bottom:1.5rem;">🎬</div>
        <h3 style="color:white; font-size:1.6rem; font-weight:800; margin:0 0 0.8rem 0;">
            Find Your Next Movie
        </h3>
        <p style="color:rgba(255,255,255,0.4); font-size:1rem; max-width:480px; margin:0 auto; line-height:1.7;">
            Search any movie from our database of <strong style="color:#a78bfa">4,806 films</strong>
            and get 5 personalized AI-powered recommendations instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2.5rem 0 1rem; color:rgba(255,255,255,0.2); font-size:0.78rem;">
    Built with ❤️ using <strong style="color:rgba(108,99,255,0.7)">Streamlit</strong> ·
    Powered by <strong style="color:rgba(108,99,255,0.7)">Cosine Similarity ML</strong> ·
    Data from <strong style="color:rgba(108,99,255,0.7)">TMDB API</strong>
</div>
""", unsafe_allow_html=True)
