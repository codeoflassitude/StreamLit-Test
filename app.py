import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("**3 Models • Feedback Loop • Content-based Recommender**")

# ====================== LOAD AND PROCESS DATA ======================
@st.cache_data(show_spinner="Loading and processing movie data...")
def load_and_process_data():
    df = pd.read_csv("tmdb_movies_filtered_500.csv")
   
    def process_genres(genre):
        if isinstance(genre, str):
            return [g.strip() for g in genre.split(', ')]
        return []
   
    def process_keywords(keywords):
        if isinstance(keywords, str):
            return [k.strip() for k in keywords.split(', ')]
        return []
   
    def keywords_to_text(keyword_list):
        if isinstance(keyword_list, list):
            cleaned = [k.strip().lower().replace(' ', '_') for k in keyword_list]
            return ' '.join(cleaned)
        return ''
   
    df['genres_list'] = df['genres'].apply(process_genres)
    df['keywords_list'] = df['keywords'].apply(process_keywords)
    df['keywords_text'] = df['keywords_list'].apply(keywords_to_text)
   
    # Multi-hot encoding for genres
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df['genres_list'])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
   
    # TF-IDF for keywords
    tfidf = TfidfVectorizer(max_features=500)
    keywords_tfidf = tfidf.fit_transform(df['keywords_text'])
    keywords_df = pd.DataFrame(keywords_tfidf.toarray(), columns=tfidf.get_feature_names_out())
   
    # Normalize numerical features
    numerical_columns = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']
    numerical_columns = [col for col in numerical_columns if col in df.columns]
    scaler = MinMaxScaler()
    df_num_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)
   
    # Weighted feature matrix
    genre_weight = 1.5
    keyword_weight = 2.0
    rating_weight = 0.5
   
    weighted_features = pd.concat([
        genres_df * genre_weight,
        keywords_df * keyword_weight,
        df_num_scaled * rating_weight
    ], axis=1)
   
    # Text for embeddings
    if 'overview' in df.columns:
        df['embed_text'] = df.apply(
            lambda row: f"Genres: {', '.join(row['genres_list'])}. Keywords: {row['keywords_text']}. Plot: {row['overview']}", axis=1)
    else:
        df['embed_text'] = df.apply(
            lambda row: f"Genres: {', '.join(row['genres_list'])}. Keywords: {row['keywords_text']}", axis=1)
   
    return df, weighted_features, tfidf, mlb.classes_

df, weighted_features, tfidf_vectorizer, genre_columns = load_and_process_data()

# ====================== LOAD MODELS ======================
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    import warnings
    from transformers import logging
    warnings.filterwarnings("ignore")
    logging.set_verbosity_error()
    return SentenceTransformer('all-MiniLM-L6-v2', token=False, trust_remote_code=True)

embedding_model = load_embedding_model()

@st.cache_data(show_spinner=False)
def get_embeddings(_df):
    return embedding_model.encode(_df['embed_text'].tolist(), batch_size=64, convert_to_numpy=True)

embeddings = get_embeddings(df)

@st.cache_resource(show_spinner=False)
def get_knn_model(_features):
    knn = NearestNeighbors(n_neighbors=30, metric='cosine', algorithm='brute')
    knn.fit(_features.values)
    return knn

knn_model = get_knn_model(weighted_features)

# ====================== SESSION STATE ======================
if 'liked_titles' not in st.session_state:
    st.session_state.liked_titles = []
if 'excluded_titles' not in st.session_state:
    st.session_state.excluded_titles = set()
if 'feedback_movies' not in st.session_state:
    st.session_state.feedback_movies = []   # Will store indices for feedback

# ====================== SIDEBAR ======================
st.sidebar.header("Your Liked Movies")
liked_input = st.sidebar.multiselect(
    "Select 3–10 movies you like",
    options=sorted(df['title'].unique()),
    default=st.session_state.liked_titles,
    max_selections=10
)

if liked_input != st.session_state.liked_titles:
    st.session_state.liked_titles = liked_input

# Display current liked movies
if st.session_state.liked_titles:
    st.sidebar.subheader("Currently Liked Movies")
    for title in st.session_state.liked_titles:
        st.sidebar.write(f"• {title}")

st.sidebar.subheader("Extra Preferences")
positive_kw = st.sidebar.text_input("Keywords to emphasize (optional)", 
                                   placeholder="e.g. dystopian ai virtual_reality")
avoid_kw = st.sidebar.text_input("Keywords to avoid (optional)", 
                                placeholder="e.g. horror comedy")

# ====================== MAIN CONTROLS ======================
st.subheader("Recommendation Settings")
model_choice = st.radio(
    "Choose Recommendation Model",
    ["Cosine Similarity", "KNN-based", "Embedding-based (Semantic)"],
    horizontal=True
)

hybrid_weight = st.slider("Hybrid popularity boost", 
                         min_value=0.0, max_value=0.3, value=0.05, step=0.05)

# ====================== GENERATE RECOMMENDATIONS ======================
if st.button("🚀 Get Recommendations", type="primary"):
    if len(st.session_state.liked_titles) < 3:
        st.error("⚠️ Please select at least 3 movies you like.")
        st.stop()

    liked_df = df[df['title'].isin(st.session_state.liked_titles)]
    
    # Generate user vector and similarity scores
    if model_choice == "Embedding-based (Semantic)":
        liked_emb = embeddings[liked_df.index]
        user_vec = liked_emb.mean(axis=0)
        if positive_kw.strip():
            extra_emb = embedding_model.encode([positive_kw.strip()])
            user_vec = (user_vec * len(liked_df) + extra_emb[0]) / (len(liked_df) + 1)
        sims = cosine_similarity([user_vec], embeddings)[0]
       
    elif model_choice == "Cosine Similarity":
        user_vec = weighted_features.loc[liked_df.index].mean(axis=0).values.reshape(1, -1)
        sims = cosine_similarity(user_vec, weighted_features.values)[0]
       
    else:  # KNN
        user_vec = weighted_features.loc[liked_df.index].mean(axis=0).values.reshape(1, -1)
        distances, indices = knn_model.kneighbors(user_vec)
        sims = 1 - distances[0]
        sims_full = np.zeros(len(df))
        sims_full[indices[0]] = sims
        sims = sims_full
   
    # Apply small hybrid boost
    pop = df['popularity'].values
    scores = sims * (1 + hybrid_weight * (pop / pop.max()))
   
    # Get top recommendations (exclude liked & excluded)
    rec_indices = np.argsort(scores)[::-1]
    filtered = []
    avoid_list = [k.strip().lower() for k in avoid_kw.split() if k.strip()] if avoid_kw else []
   
    for i in rec_indices:
        title = df.iloc[i]['title']
        if title in st.session_state.liked_titles or title in st.session_state.excluded_titles:
            continue
        if avoid_list and any(k in str(df.iloc[i]['keywords_text']).lower() for k in avoid_list):
            continue
        filtered.append(i)
        if len(filtered) >= 15:
            break
   
    st.session_state.rec_df = df.iloc[filtered][['title', 'genres', 'vote_average', 'popularity', 'keywords_text']].copy()
    st.session_state.rec_df['score'] = [scores[i] for i in filtered]
    st.session_state.rec_df = st.session_state.rec_df.head(15)
    
    st.success(f"✅ Top 15 recommendations using **{model_choice}**")
    
    # Display recommendations
    display_df = st.session_state.rec_df[['title', 'score', 'genres', 'vote_average', 'popularity']].copy()
    display_df = display_df.style.format({
        'score': "{:.3f}",
        'vote_average': "{:.1f}",
        'popularity': "{:.1f}"
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ====================== FEEDBACK LOOP (Random Movies) ======================
st.subheader("💬 Feedback Loop - Discover & Rate Random Movies")
st.caption("Rate random movies to better understand your taste. Click 'Refresh Recommendations' when done.")

if st.button("🎲 Show Random Movies for Feedback"):
    # Fix: Convert liked_titles (list) to set for proper union
    liked_set = set(st.session_state.liked_titles)
    excluded_set = st.session_state.excluded_titles
    
    # Generate 12 random movies (excluding already liked or excluded)
    available = df[~df['title'].isin(liked_set | excluded_set)]
    
    if len(available) == 0:
        st.warning("No more movies available for feedback.")
    else:
        n = min(12, len(available))
        random_indices = random.sample(range(len(available)), n)
        st.session_state.feedback_movies = available.iloc[random_indices].index.tolist()
        st.success(f"✅ Showing {n} random movies for feedback")

# Show feedback movies if available
if st.session_state.get('feedback_movies'):
    st.write("Rate these movies:")
    for idx in st.session_state.feedback_movies:
        row = df.iloc[idx]
        col1, col2, col3 = st.columns([6, 1, 1])
        with col1:
            st.write(f"**{row['title']}**")
            st.caption(f"Genres: {row.get('genres', 'N/A')} | Rating: {row.get('vote_average', 0):.1f}")
        with col2:
            if st.button("👍 Like", key=f"like_{idx}"):
                if row['title'] not in st.session_state.liked_titles:
                    st.session_state.liked_titles.append(row['title'])
                    st.toast(f"Added '{row['title']}' to liked movies")
        with col3:
            if st.button("👎 Dislike", key=f"dislike_{idx}"):
                st.session_state.excluded_titles.add(row['title'])
                st.toast(f"Excluded '{row['title']}'")

# ====================== REFRESH BUTTON ======================
if st.button("🔄 Refresh Recommendations", type="secondary"):
    st.rerun()   # This will re-run the whole script and show updated liked movies + new recommendations

st.caption("Liked movies are automatically used for the next recommendation round.")
