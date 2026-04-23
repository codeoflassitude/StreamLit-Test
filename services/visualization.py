import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def get_pca_projection(movie_features_df):
    """Project feature vectors into 2D using PCA."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(movie_features_df.values)
    return pd.DataFrame(coords, columns=['x', 'y'])


def build_visualization_figure(df, movie_features_df, coords, liked_indices, rec_indices, model_type, rec_scores=None):
    """Build model-specific 2D PCA visualization."""
    viz_df = df[['title']].copy()
    viz_df['x'] = coords['x']
    viz_df['y'] = coords['y']
    viz_df['Type'] = 'Other Movies'
    viz_df.loc[liked_indices, 'Type'] = 'Liked Movies'
    viz_df.loc[rec_indices, 'Type'] = 'Recommended Movies'

    color_map = {
        'Other Movies': '#C9CED6',
        'Liked Movies': '#2563EB',
        'Recommended Movies': '#DC2626'
    }

    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='Type',
        color_discrete_map=color_map,
        hover_data={'title': True, 'x': False, 'y': False},
        title=f'{model_type} - 2D Movie Feature Space (PCA)',
        opacity=0.8
    )
    fig.update_traces(marker=dict(size=7), selector=dict(mode='markers'))

    if model_type == "Cosine Similarity":
        liked_coords = coords.iloc[liked_indices]
        centroid_x = float(liked_coords['x'].mean())
        centroid_y = float(liked_coords['y'].mean())

        fig.add_trace(go.Scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode='markers',
            marker=dict(symbol='star', size=16, color='#F59E0B', line=dict(color='black', width=1)),
            name='User Taste Centroid',
            hovertemplate='User Taste Centroid<extra></extra>'
        ))

        for i, rec_idx in enumerate(rec_indices):
            x1, y1 = coords.iloc[int(rec_idx)][['x', 'y']]
            line_width = 1.5
            if rec_scores is not None and len(rec_scores) > i:
                line_width = 1 + (float(rec_scores[i]) * 2.5)

            fig.add_trace(go.Scatter(
                x=[centroid_x, x1],
                y=[centroid_y, y1],
                mode='lines',
                line=dict(color='rgba(245,158,11,0.45)', width=line_width),
                hoverinfo='skip',
                showlegend=False
            ))
    else:
        features = movie_features_df.values
        for rec_idx in rec_indices:
            rec_vec = features[int(rec_idx)].reshape(1, -1)
            liked_vecs = features[liked_indices]
            sim_to_liked = cosine_similarity(rec_vec, liked_vecs)[0]
            nearest_pos = int(sim_to_liked.argmax())
            nearest_liked_idx = liked_indices[nearest_pos]

            x0, y0 = coords.iloc[int(rec_idx)][['x', 'y']]
            x1, y1 = coords.iloc[int(nearest_liked_idx)][['x', 'y']]

            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color='rgba(37,99,235,0.45)', width=1.8),
                hoverinfo='skip',
                showlegend=False
            ))

    fig.update_layout(legend_title_text='Point Type')
    return fig


def build_overlap_figure(df, coords, liked_indices, cosine_rec_indices, knn_rec_indices):
    """Build comparison plot showing overlap and differences between models."""
    viz_df = df[['title']].copy()
    viz_df['x'] = coords['x']
    viz_df['y'] = coords['y']
    viz_df['Type'] = 'Other Movies'

    cosine_set = set(map(int, cosine_rec_indices))
    knn_set = set(map(int, knn_rec_indices))
    both_set = cosine_set.intersection(knn_set)
    cosine_only = cosine_set.difference(knn_set)
    knn_only = knn_set.difference(cosine_set)

    viz_df.loc[liked_indices, 'Type'] = 'Liked Movies'
    if both_set:
        viz_df.loc[list(both_set), 'Type'] = 'Recommended by Both'
    if cosine_only:
        viz_df.loc[list(cosine_only), 'Type'] = 'Cosine Only'
    if knn_only:
        viz_df.loc[list(knn_only), 'Type'] = 'KNN Only'

    color_map = {
        'Other Movies': '#D1D5DB',
        'Liked Movies': '#2563EB',
        'Recommended by Both': '#7C3AED',
        'Cosine Only': '#F59E0B',
        'KNN Only': '#16A34A'
    }

    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='Type',
        color_discrete_map=color_map,
        hover_data={'title': True, 'x': False, 'y': False},
        title='Model Comparison: Overlap vs Differences',
        opacity=0.85
    )
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    fig.update_layout(legend_title_text='Recommendation Source')
    return fig
