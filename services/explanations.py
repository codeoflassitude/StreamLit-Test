import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _split_csv_text(value):
    if not isinstance(value, str):
        return []
    return [v.strip().lower() for v in value.split(',') if v.strip()]


def build_recommendation_explanations(df, movie_features_df, rec_indices, liked_indices, scores, model_name):
    """Create human-readable explanation per recommended movie."""
    explanations = {}
    features = movie_features_df.values
    
    # Ensure scores is a numpy array
    scores = np.asarray(scores).flatten()

    liked_genres_union = set()
    liked_keywords_union = set()

    for li in liked_indices:
        liked_genres_union.update(_split_csv_text(df.iloc[li].get('genres', '')))
        liked_keywords_union.update(_split_csv_text(df.iloc[li].get('keywords', '')))

    for i, rec_idx in enumerate(rec_indices):
        rec_vec = features[int(rec_idx)].reshape(1, -1)
        liked_vecs = features[liked_indices]

        sim_to_liked = cosine_similarity(rec_vec, liked_vecs)[0]
        nearest_pos = int(np.argmax(sim_to_liked))
        nearest_liked_idx = liked_indices[nearest_pos]
        nearest_liked_title = df.iloc[nearest_liked_idx]['title']
        nearest_similarity = float(sim_to_liked[nearest_pos])

        rec_genres = set(_split_csv_text(df.iloc[int(rec_idx)].get('genres', '')))
        rec_keywords = set(_split_csv_text(df.iloc[int(rec_idx)].get('keywords', '')))

        shared_genres = sorted(list(rec_genres.intersection(liked_genres_union)))[:3]
        shared_keywords = sorted(list(rec_keywords.intersection(liked_keywords_union)))[:5]

        # Update summary based on model name
        if model_name == "Content-Based":
            summary = (
                f"High feature similarity to your taste profile. "
                f"Closest to your liked movie '{nearest_liked_title}'."
            )
        elif model_name == "Semantic Embedding":
            summary = (
                f"Thematically similar to your favorite movies. "
                f"Shares narrative elements with '{nearest_liked_title}'."
            )
        else:  # Popularity Hybrid
            summary = (
                f"Well-rated movie matching your preferences. "
                f"Similar to '{nearest_liked_title}'."
            )

        # Safely access score with bounds checking
        score_value = float(scores[i]) if i < len(scores) else 0.0
        
        explanations[int(rec_idx)] = {
            'summary': summary,
            'nearest_title': nearest_liked_title,
            'nearest_similarity': round(nearest_similarity * 100, 1),
            'shared_genres': shared_genres,
            'shared_keywords': shared_keywords,
            'model_score': float(f"{score_value * 100:.1f}")
        }

    return explanations
