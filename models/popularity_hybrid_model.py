import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class PopularityHybridModel:
    """
    Hybrid Recommender combining Content Similarity with Popularity Weighting.
    
    Algorithm: Weighted blend of content-based filtering + external popularity signal
    
    How it works:
    - SCORE A - Content Similarity:
      * Uses same 556-dim feature space as ContentBasedModel
      * Computes cosine_similarity(user_profile, all_movies)
      * Normalizes to [0, 1] range
    
    - SCORE B - Popularity/Quality Signal:
      * rating_norm = vote_average / 10.0 ∈ [0, 1]
      * vote_norm = (vote_count - min) / (max - min) ∈ [0, 1]
      * quality_score = 0.6 × rating_norm + 0.4 × vote_norm (emphasizes ratings)
      * popularity_norm = (popularity - min) / (max - min) ∈ [0, 1]
    
    - BLENDING:
      * hybrid_score = 0.7 × content_similarity + 0.3 × popularity
      * 70% weight: personalization (similarity to your taste)
      * 30% weight: popularity (external quality signal)
    
    - Returns: Top N movies by hybrid_score
    
    Design Philosophy:
    - Balances exploration (discovering popular movies) with exploitation (matching your taste)
    - Reduces risk of recommending low-quality or unknown movies
    - Weights tunable for different recommendation profiles
    
    Strengths:
    - Combines personal taste with objective quality metrics
    - Discovers well-rated mainstream movies matching your profile
    - Smooth interpolation between pure content and pure popularity
    - Still relatively fast (no model training)
    
    Limitations:
    - Popularity bias: tends to favor older/bigger-budget films
    - Less personalization than pure content-based
    - Weighting ratios (70/30) are hardcoded, not learned
    - Cannot explain why movies are popular to specific users
    
    Normalization Details:
    - All scores normalized independently to [0, 1] before blending
    - Prevents one signal from dominating due to different scales
    - Handles edge cases: divide-by-zero protected with 1e-6 epsilon
    """

    def __init__(self, features_df, df):
        """
        Initialize hybrid model.
        
        Args:
            features_df: Feature vectors for movies
            df: Movie metadata DataFrame with 'popularity' and 'vote_average' columns
        """
        self.features = features_df.values
        self.df = df
        
        # Normalize popularity score to [0, 1]
        self.popularity_norm = (df['popularity'].fillna(0) - df['popularity'].min()) / (df['popularity'].max() - df['popularity'].min() + 1e-6)
        
        # Quality score: weighted average of rating and vote count
        vote_norm = (df['vote_count'].fillna(0) - df['vote_count'].min()) / (df['vote_count'].max() - df['vote_count'].min() + 1e-6)
        rating_norm = df['vote_average'].fillna(0) / 10.0
        self.quality_score = 0.6 * rating_norm + 0.4 * vote_norm

    def recommend(self, liked_indices, n_recommendations=5):
        """Generate recommendations using hybrid content + popularity scoring."""
        liked_indices = np.array(liked_indices, dtype=int)
        if liked_indices.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        n_recommendations = max(1, int(n_recommendations))

        # Content-based similarity
        liked_features = self.features[liked_indices]
        user_vector = np.mean(liked_features, axis=0).reshape(1, -1)
        content_similarities = cosine_similarity(user_vector, self.features)[0]
        
        # Normalize content similarity to [0, 1]
        content_similarities_norm = (content_similarities - content_similarities.min()) / (content_similarities.max() - content_similarities.min() + 1e-6)

        # Hybrid scoring: 70% content, 30% popularity
        hybrid_scores = 0.7 * content_similarities_norm + 0.3 * self.popularity_norm

        # Exclude liked movies
        hybrid_scores[liked_indices] = -1

        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]
        return top_indices, hybrid_scores[top_indices]
