import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd


class EmbeddingBasedModel:
    """
    Semantic Embedding-Based Recommender using Sentence-Transformers.
    
    Algorithm: Neural network-based semantic similarity
    
    How it works:
    - Uses pre-trained Sentence-Transformers model (all-MiniLM-L6-v2, 138M parameters)
    - Encodes movie plot overviews into 384-dimensional semantic embeddings
    - Embeddings capture: story themes, narrative patterns, emotional tone, plot devices
    - Creates user profile: user_embedding = mean(liked_movies_embeddings)
    - Computes cosine similarity in embedding space
    - Returns top N movies by semantic similarity
    
    Neural Network Details:
    - Base: BERT-style Transformer architecture (12 layers, 256 hidden dims, 4 attention heads)
    - Pre-training: 215M+ sentence pair datasets (NLI, paraphrase, clustering tasks)
    - Output: 384-dimensional vectors with unit normalization (L2)
    - Transfer Learning: Leverages knowledge from diverse NLP tasks
    
    Feature Extraction:
    - Processes movie overview text (plot summaries)
    - Handles variable-length input (adaptive pooling)
    - Captures: plot devices, genres (implicit), tone, character types, themes
    
    Strengths:
    - Discovers thematically similar movies even without shared keywords
    - Generalizes well to new plots and story types
    - Captures semantic meaning and narrative patterns
    - More "human-like" recommendations
    
    Limitations:
    - Slower: ~1-2 seconds for initial encoding (subsequent lookups are fast)
    - Depends heavily on plot description quality and length
    - Less interpretable than feature-based approaches
    - Requires sentence-transformers library and computational resources
    
    Computational Notes:
    - Time: O(n) initial encoding, O(1) per new recommendation
    - Space: O(n × 384) for storing embeddings
    - Model size: ~80MB (downloaded on first run, cached locally)
    """

    def __init__(self, df, model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding model.
        
        Args:
            df: DataFrame with 'overview' column
            model_name: Sentence-Transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.df = df
        self._embeddings = None
        self._generate_embeddings()

    def _generate_embeddings(self):
        """Generate embeddings for all movie overviews."""
        overviews = self.df['overview'].fillna("").tolist()
        self._embeddings = self.model.encode(overviews, convert_to_numpy=True)

    def recommend(self, liked_indices, n_recommendations=5):
        """Generate recommendations based on semantic similarity of overviews."""
        liked_indices = np.array(liked_indices, dtype=int)
        if liked_indices.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        n_recommendations = max(1, int(n_recommendations))

        # Create user profile from liked movie embeddings
        liked_embeddings = self._embeddings[liked_indices]
        user_embedding = np.mean(liked_embeddings, axis=0).reshape(1, -1)

        # Compute similarities
        similarities = cosine_similarity(user_embedding, self._embeddings)[0]
        similarities[liked_indices] = -1

        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        return top_indices, similarities[top_indices]
