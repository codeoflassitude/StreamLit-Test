import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from models.content_based_model import ContentBasedModel
from models.embedding_model import EmbeddingBasedModel
from models.popularity_hybrid_model import PopularityHybridModel
from services.explanations import build_recommendation_explanations
from services.recommendation_utils import (
    analyze_user_taste,
    format_recommendations,
    get_movie_suggestions,
)
from services.visualization import (
    build_overlap_figure,
    build_visualization_figure,
    get_pca_projection,
)

# Set page config
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    """Load movie data and feature vectors"""
    df = pd.read_csv("tmdb_movies_filtered_500.csv")
    movie_features = pd.read_csv("movie_feature_vectors.csv")
    return df, movie_features

df, movie_features = load_data()

# Initialize session state
if 'liked_movies' not in st.session_state:
    st.session_state.liked_movies = []

if 'results' not in st.session_state:
    st.session_state.results = None

@st.cache_data
def get_cached_projection(features_df):
    """Cache PCA projection for visualization."""
    return get_pca_projection(features_df)


coords = get_cached_projection(movie_features)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("🎬 Movie Recommendation System")
st.markdown("Find your next favorite movie based on your preferences!")

# ============================================================================
# SIDEBAR - MOVIE SELECTION
# ============================================================================

with st.sidebar:
    st.header("Step 1: Select Your Favorite Movies")
    
    # Search input
    search_query = st.text_input(
        "🔍 Search for a movie:",
        placeholder="Type movie name..."
    )
    
    # Show suggestions
    if search_query and len(search_query) > 1:
        suggestions = get_movie_suggestions(df, search_query)
        
        if suggestions:
            selected_movie = st.selectbox(
                "Select from suggestions:",
                suggestions,
                key="movie_selector"
            )
            
            if st.button("➕ Add Movie"):
                movie_idx = df[df['title'] == selected_movie].index[0]
                
                if selected_movie not in st.session_state.liked_movies:
                    if len(st.session_state.liked_movies) < 5:
                        st.session_state.liked_movies.append(selected_movie)
                        st.success(f"✓ Added: {selected_movie}")
                    else:
                        st.error("⚠️ Maximum 5 movies selected!")
                else:
                    st.warning("Movie already in your list!")
    
    # Display selected movies
    st.markdown("---")
    st.subheader(f"Your Selection ({len(st.session_state.liked_movies)}/5)")
    
    for i, movie in enumerate(st.session_state.liked_movies, 1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{i}. {movie}")
        with col2:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.liked_movies.pop(i-1)
                st.session_state.results = None
                st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

if len(st.session_state.liked_movies) >= 3:
    st.markdown("---")
    
    # Step 2: Model Selection
    st.header("Step 2: Choose Recommendation Model")
    
    model_choice = st.radio(
        "Select a model:",
        ["Content-Based", "Semantic Embedding", "Popularity Hybrid"],
        help="Choose how movies should be recommended to you"
    )
    
    # Show model explanation with technical details
    model_explanations = {
        "Content-Based": {
            "icon": "📋",
            "title": "Content-Based Filtering",
            "desc": "Analyzes genres, keywords, and numerical features of your favorite movies to find similar ones.",
            "technical": """
**Algorithm**: Feature Averaging + Cosine Similarity

**How it works**:
1. Extracts 556 features per movie: 50 genre indicators (one-hot), 500 TF-IDF keyword scores, 6 numerical metrics (rating, popularity, budget, etc.)
2. Creates user profile: `user_vector = average(liked_movies_features)`
3. Computes cosine similarity: `similarity = (user·movie) / (||user|| × ||movie||)` for all movies
4. Returns top recommendations by similarity score

**Pros**: Fast (< 100ms), transparent, no data requirements
**Cons**: Misses semantic meaning, can be too specific to liked movie features
            """
        },
        "Semantic Embedding": {
            "icon": "🧠",
            "title": "Semantic Embedding (AI-Powered)",
            "desc": "Uses neural networks to understand movie plots and themes. Finds movies with similar stories even if they don't share keywords.",
            "technical": """
**Algorithm**: Sentence-Transformers (BERT-based) + Cosine Similarity

**How it works**:
1. Uses pre-trained `all-MiniLM-L6-v2` model (138M parameters) to encode movie overviews
2. Each plot → 384-dimensional semantic embedding (captures meaning, themes, narrative patterns)
3. Creates user profile: `user_embedding = average(liked_movies_embeddings)`
4. Computes cosine similarity in embedding space for all movies
5. Returns top recommendations by semantic similarity

**Architecture**: Transformer-based (BERT-variant), trained on 215M+ sentence pairs for semantic understanding

**Pros**: Finds thematically similar movies, discovers hidden connections, generalizes well
**Cons**: Slower (~1-2s initial encoding), depends on plot description quality, requires pre-trained model
            """
        },
        "Popularity Hybrid": {
            "icon": "⭐",
            "title": "Popularity Hybrid",
            "desc": "Balances personal taste with movie popularity. Recommends well-rated movies that match your preferences.",
            "technical": """
**Algorithm**: Weighted Blend of Content Similarity + Popularity Metrics

**How it works**:
1. Computes content similarity (same as Content-Based): user_vector averaged from liked movies
2. Normalizes content scores to [0, 1] range
3. Calculates popularity score: `popularity_norm = (popularity - min) / (max - min)`
4. Calculates quality score: `0.6 × (rating/10) + 0.4 × vote_count_norm`
5. Blends: `hybrid_score = 0.7 × content_similarity + 0.3 × popularity`
6. Returns top recommendations by hybrid score

**Weighting**: 70% personalization, 30% popularity (tunable)

**Pros**: Balances niche with mainstream, reduces risk of recommending obscure low-quality films
**Cons**: May favor popular movies even if less similar to your taste
            """
        }
    }
    
    exp = model_explanations[model_choice]
    st.info(f"**{exp['icon']} {exp['title']}**: {exp['desc']}")
    
    with st.expander("📊 Technical Details"):
        st.markdown(exp['technical'])

    n_recommendations = 5
    
    # Generate recommendations
    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing your taste and generating recommendations..."):
            # Get indices of liked movies
            liked_indices = [df[df['title'] == m].index[0] for m in st.session_state.liked_movies]
            
            # Analyze user taste
            taste_profile = analyze_user_taste(df, liked_indices)
            
            # Generate recommendations for all models (for comparison visualization)
            content_model = ContentBasedModel(movie_features)
            content_rec_indices, content_scores = content_model.recommend(liked_indices, n_recommendations)

            embedding_model = EmbeddingBasedModel(df)
            embedding_rec_indices, embedding_scores = embedding_model.recommend(liked_indices, n_recommendations)

            hybrid_model = PopularityHybridModel(movie_features, df)
            hybrid_rec_indices, hybrid_scores = hybrid_model.recommend(liked_indices, n_recommendations)

            # Active model for main recommendations tab
            model_map = {
                "Content-Based": (content_rec_indices, content_scores),
                "Semantic Embedding": (embedding_rec_indices, embedding_scores),
                "Popularity Hybrid": (hybrid_rec_indices, hybrid_scores)
            }
            rec_indices, scores = model_map[model_choice]

            explanations = build_recommendation_explanations(
                df=df,
                movie_features_df=movie_features,
                rec_indices=rec_indices,
                liked_indices=liked_indices,
                scores=scores,
                model_name=model_choice
            )

            st.session_state.results = {
                'model_choice': model_choice,
                'liked_indices': liked_indices,
                'rec_indices': rec_indices,
                'scores': scores,
                'content_rec_indices': content_rec_indices,
                'content_scores': content_scores,
                'embedding_rec_indices': embedding_rec_indices,
                'embedding_scores': embedding_scores,
                'hybrid_rec_indices': hybrid_rec_indices,
                'hybrid_scores': hybrid_scores,
                'taste_profile': taste_profile,
                'recs_df': format_recommendations(df, rec_indices, scores),
                'explanations': explanations
            }

    if st.session_state.results is not None:
        results = st.session_state.results
        st.markdown("---")

        tab1, tab2 = st.tabs(["Recommendations", "Why These Movies"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Movies Analyzed", results['taste_profile']['num_movies'])
            with col2:
                st.metric("Avg Rating", f"{results['taste_profile']['avg_rating']:.1f}/10")
            with col3:
                st.metric("Avg Popularity", f"{results['taste_profile']['avg_popularity']:.0f}")

            st.subheader("📊 Your Taste Profile")
            fig = px.pie(
                values=results['taste_profile']['genres'].values,
                names=results['taste_profile']['genres'].index,
                title="Genre Distribution of Your Favorites"
            )
            st.plotly_chart(fig, use_container_width=True)

            mood_col, time_col = st.columns(2)

            with mood_col:
                st.markdown("**Mood themes from your liked movies**")
                top_keywords = results['taste_profile'].get('top_keywords')
                if top_keywords is not None and len(top_keywords) > 0:
                    mood_line = " • ".join([kw.title() for kw in top_keywords.index[:6]])
                    st.write(mood_line)
                else:
                    st.caption("Not enough keyword data to infer mood themes.")

            with time_col:
                st.markdown("**Your time preference**")
                avg_year = results['taste_profile'].get('avg_year')
                favorite_decade = results['taste_profile'].get('favorite_decade')
                if favorite_decade is not None:
                    if avg_year is not None:
                        st.write(f"Mostly {favorite_decade}s films • Average release year: {avg_year:.0f}")
                    else:
                        st.write(f"Mostly {favorite_decade}s films")
                else:
                    st.caption("Release year data is limited for this selection.")

            st.subheader(f"🎯 Recommended Movies ({results['model_choice']})")
            rec_indices_array = np.asarray(results['rec_indices']).flatten()
            for pos, (_, row) in enumerate(results['recs_df'].iterrows()):
                rec_idx = int(rec_indices_array[pos])
                exp = results['explanations'][rec_idx]

                with st.container(border=True):
                    poster_col, info_col = st.columns([1, 3])

                    with poster_col:
                        poster_url = row.get('Poster URL')
                        if isinstance(poster_url, str) and poster_url:
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.caption("Poster unavailable")

                    with info_col:
                        st.markdown(f"### {row['Title']}")
                        st.caption(f"Release Date: {row['Release Date']}")

                        score_col1, score_col2 = st.columns(2)
                        with score_col1:
                            st.metric("IMDb Rating", f"{row['IMDb Rating']}/10")
                        with score_col2:
                            st.metric("Match Score", f"{row['similarity_score']}%")

                        st.write(f"**Genres:** {row['Genres']}")
                        st.write(f"**Language:** {row['Language']}")
                        st.write(f"**Description:** {row['Description']}")
                        st.info(exp['summary'])

        with tab2:
            st.subheader("🔍 Why each movie was recommended")
            rec_indices_array = np.asarray(results['rec_indices']).flatten()
            for pos, (_, row) in enumerate(results['recs_df'].iterrows()):
                rec_idx = int(rec_indices_array[pos])
                exp = results['explanations'][rec_idx]

                with st.container(border=True):
                    st.markdown(f"### {row['Title']}")
                    st.write(f"**Model score:** {exp['model_score']}%")
                    st.write(f"**Closest liked movie:** {exp['nearest_title']} ({exp['nearest_similarity']}% similarity)")

                    shared_genres_text = ', '.join(exp['shared_genres']) if exp['shared_genres'] else 'None detected'
                    shared_keywords_text = ', '.join(exp['shared_keywords']) if exp['shared_keywords'] else 'None detected'

                    st.write(f"**Shared genres:** {shared_genres_text}")
                    st.write(f"**Shared keywords:** {shared_keywords_text}")
                    st.write(f"**Why chosen:** {exp['summary']}")
            
else:
    # Placeholder when not enough movies selected
    st.info(f"👉 Select at least 3 favorite movies to get started! ({len(st.session_state.liked_movies)}/3)")
    
    st.markdown("""
    ### How it works:
    1. **Search & Select:** Find your favorite movies using the search bar
    2. **Choose Model:** Pick between Cosine Similarity or KNN
    3. **Get Recommendations:** See personalized movie suggestions with analysis!
    """)
