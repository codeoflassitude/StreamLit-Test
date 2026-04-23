import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedModel:
    """
    Content-Based Filtering using Feature Averaging and Cosine Similarity.
    
    Algorithm: Explicit feature matching in handcrafted feature space
    
    How it works:
    - Extracts 556 features per movie: 50 genre one-hot encodings, 500 TF-IDF keyword scores,
      6 numerical metrics (vote_average, vote_count, revenue, runtime, budget, popularity)
    - Normalizes all numerical features to [0, 1] range
    - Creates user profile: user_vector = mean(liked_movies_features)
    - Computes cosine similarity: similarity = (user·movie) / (||user|| × ||movie||) ∈ [-1, 1]
    - Returns top N movies by similarity score, excluding liked movies
    
    Feature Engineering:
    - Genres: Multi-hot encoding (1 if present, 0 otherwise)
    - Keywords: TF-IDF vectorization with 500 dimensions
    - Numerical: Log-scale transform (for skewed features) → MinMax normalization
    - Weights: genre×1.5, keyword×2.0, numerical×0.5
    
    Strengths:
    - O(n) complexity, fast recommendations (< 100ms)
    - Completely transparent and interpretable
    - No training required, deterministic
    - Works well with limited data
    
    Limitations:
    - Operates in fixed feature space, misses semantic meaning
    - Cannot discover unexpected/serendipitous recommendations
    - Affected by feature engineering choices
    """

    def __init__(self, features_df):
        self.features = features_df.values

    def recommend(self, liked_indices, n_recommendations=5):
        """Generate recommendations using cosine similarity to user profile vector."""
        liked_indices = np.array(liked_indices, dtype=int)
        if liked_indices.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        n_recommendations = max(1, int(n_recommendations))

        liked_features = self.features[liked_indices]
        user_vector = np.mean(liked_features, axis=0).reshape(1, -1)

        similarities = cosine_similarity(user_vector, self.features)[0]
        similarities[liked_indices] = -1

        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        return top_indices, similarities[top_indices]
