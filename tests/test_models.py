"""
Tests unitaires pour les fonctions de modélisation.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from src.models.pca import compute_pca
from src.models.clustering import compute_elbow_curve, train_kmeans
from src.models.anomaly_detection import train_isolation_forest


@pytest.fixture
def sample_data():
    """Crée des données de test pour les modèles."""
    np.random.seed(42)
    X, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)
    feature_names = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    return df


def test_compute_pca(sample_data):
    """Test la fonction PCA."""
    pca_result = compute_pca(sample_data, n_components=3)
    
    # Vérifications de base
    assert 'X_pca' in pca_result
    assert 'explained_variance_ratio' in pca_result
    assert 'pca' in pca_result
    assert 'scaler' in pca_result
    
    # Vérifier les dimensions
    assert pca_result['X_pca'].shape[0] == len(sample_data)
    assert pca_result['X_pca'].shape[1] == 3
    
    # Vérifier la variance expliquée
    explained = pca_result['explained_variance_ratio']
    assert len(explained) == 3
    assert np.all(explained >= 0)
    assert np.all(explained <= 1)
    
    # Vérifier la variance cumulée
    cumulative = pca_result.get('cumulative_variance', np.cumsum(explained))
    assert cumulative[-1] <= 1.0


def test_compute_elbow_curve(sample_data):
    """Test la courbe du coude."""
    ks, inertias = compute_elbow_curve(sample_data, k_min=2, k_max=5)
    
    # Vérifications de base
    assert len(ks) == 4  # 2, 3, 4, 5
    assert len(inertias) == 4
    
    # Vérifier que ks est dans l'ordre
    assert list(ks) == [2, 3, 4, 5]
    
    # Vérifier que l'inertie diminue quand k augmente
    for i in range(len(inertias) - 1):
        assert inertias[i] > inertias[i + 1], f"Inertia should decrease with k, got {inertias[i]} <= {inertias[i+1]}"


def test_train_kmeans(sample_data):
    """Test l'entraînement KMeans."""
    kmeans_result = train_kmeans(sample_data, n_clusters=3)
    
    # Vérifications de base
    assert 'kmeans' in kmeans_result
    assert 'cluster_labels' in kmeans_result
    assert 'silhouette_score' in kmeans_result
    assert 'X_transformed' in kmeans_result
    
    # Vérifier les dimensions
    assert len(kmeans_result['cluster_labels']) == len(sample_data)
    assert kmeans_result['X_transformed'].shape[0] == len(sample_data)
    
    # Vérifier le nombre de clusters
    unique_labels = np.unique(kmeans_result['cluster_labels'])
    assert len(unique_labels) == 3
    
    # Vérifier le score silhouette
    silhouette = kmeans_result['silhouette_score']
    assert -1 <= silhouette <= 1


def test_train_isolation_forest(sample_data):
    """Test l'entraînement Isolation Forest."""
    iforest_result = train_isolation_forest(sample_data, contamination=0.1)
    
    # Vérifications de base
    assert 'iforest' in iforest_result
    assert 'anomaly_scores' in iforest_result
    assert 'is_anomaly' in iforest_result
    assert 'X_scaled' in iforest_result
    
    # Vérifier les dimensions
    assert len(iforest_result['anomaly_scores']) == len(sample_data)
    assert len(iforest_result['is_anomaly']) == len(sample_data)
    assert iforest_result['X_scaled'].shape[0] == len(sample_data)
    
    # Vérifier le taux d'anomalies
    anomaly_rate = iforest_result['is_anomaly'].mean()
    assert abs(anomaly_rate - 0.1) < 0.05  # Tolérance de 5%
    
    # Vérifier les scores
    scores = iforest_result['anomaly_scores']
    assert scores.min() >= 0  # Scores normalisés sont positifs


def test_pca_auto_components(sample_data):
    """Test la détermination automatique du nombre de composantes."""
    pca_result = compute_pca(sample_data, n_components=None, variance_threshold=0.8)
    
    # Vérifier qu'un nombre de composantes a été déterminé
    assert 'n_components' in pca_result
    assert pca_result['n_components'] >= 1
    assert pca_result['n_components'] <= sample_data.shape[1]
    
    # Vérifier que la variance cumulée atteint le seuil
    cumulative = pca_result['cumulative_variance'][-1]
    assert cumulative >= 0.8


def test_kmeans_pca_option(sample_data):
    """Test l'option PCA dans KMeans."""
    # Avec PCA
    kmeans_with_pca = train_kmeans(sample_data, n_clusters=3, use_pca=True, n_components=2)
    assert kmeans_with_pca['pca'] is not None
    assert kmeans_with_pca['X_transformed'].shape[1] == 2
    
    # Sans PCA
    kmeans_without_pca = train_kmeans(sample_data, n_clusters=3, use_pca=False)
    assert kmeans_without_pca['pca'] is None
    assert kmeans_without_pca['X_transformed'].shape[1] == sample_data.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])