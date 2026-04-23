from difflib import get_close_matches

import numpy as np
import pandas as pd


TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"


def get_movie_suggestions(df, search_query):
    """Get autocomplete title suggestions."""
    movie_list = df['title'].tolist()
    return get_close_matches(search_query, movie_list, n=10, cutoff=0.3)


def analyze_user_taste(df, liked_indices):
    """Analyze user taste profile from liked movies."""
    liked_movies = df.iloc[liked_indices]

    all_genres = []
    for genres_str in liked_movies['genres']:
        if isinstance(genres_str, str):
            all_genres.extend([g.strip() for g in genres_str.split(',')])

    genre_counts = pd.Series(all_genres).value_counts()

    # Mood/theme from keywords
    all_keywords = []
    if 'keywords' in liked_movies.columns:
        for keywords_str in liked_movies['keywords']:
            if isinstance(keywords_str, str):
                all_keywords.extend([k.strip().lower() for k in keywords_str.split(',') if k.strip()])
    keyword_counts = pd.Series(all_keywords).value_counts() if all_keywords else pd.Series(dtype=int)

    # Time preference from release years
    release_years = pd.to_datetime(liked_movies['release_date'], errors='coerce').dt.year
    release_years = release_years.dropna().astype(int)
    decade_counts = pd.Series(dtype=int)
    avg_year = None
    favorite_decade = None
    if not release_years.empty:
        decades = (release_years // 10) * 10
        decade_counts = decades.value_counts().sort_index()
        avg_year = float(release_years.mean())
        favorite_decade = int(decades.mode().iloc[0])

    return {
        'genres': genre_counts,
        'top_keywords': keyword_counts.head(6),
        'avg_year': avg_year,
        'favorite_decade': favorite_decade,
        'decade_counts': decade_counts,
        'avg_rating': liked_movies['vote_average'].mean(),
        'avg_popularity': liked_movies['popularity'].mean(),
        'num_movies': len(liked_indices)
    }


def format_recommendations(df, rec_indices, scores):
    """Format recommendation rows for UI rendering."""
    language_col = 'original_language' if 'original_language' in df.columns else None
    if language_col is None and 'spoken_languages' in df.columns:
        language_col = 'spoken_languages'

    base_columns = ['title', 'vote_average', 'genres', 'overview', 'release_date']
    if language_col:
        base_columns.append(language_col)
    if 'poster_path' in df.columns:
        base_columns.append('poster_path')

    recs = df.iloc[rec_indices][base_columns].copy()

    if language_col is None:
        recs['Language'] = 'N/A'
    elif language_col == 'original_language':
        recs['Language'] = recs[language_col].fillna('N/A').str.upper()
    else:
        recs['Language'] = recs[language_col].fillna('N/A')

    recs['similarity_score'] = [float(f"{s:.1f}") for s in scores * 100]

    if 'poster_path' in recs.columns:
        recs['Poster URL'] = recs['poster_path'].apply(
            lambda p: f"{TMDB_IMAGE_BASE_URL}{p}" if isinstance(p, str) and p.strip() and p != 'nan' else None
        )
    else:
        recs['Poster URL'] = None

    recs = recs.rename(columns={
        'title': 'Title',
        'vote_average': 'IMDb Rating',
        'genres': 'Genres',
        'overview': 'Description',
        'release_date': 'Release Date'
    })

    return recs
