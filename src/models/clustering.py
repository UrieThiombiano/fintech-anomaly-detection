"""
Fonctions pour le clustering (KMeans).
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from src.logging_config import get_logger
from src.utils import scale_features

logger = get_logger(__name__)

def compute_elbow_curve(
    X: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> Tuple[List[int], List[float]]:
    """
    Calcule la courbe du coude pour KMeans.
    """
    logger.info(f"Calcul de la courbe du coude pour k de {k_min} à {k_max}")
    
    # Préparer les données
    X_clean = prepare_data_for_clustering(X)
    
    # Standardisation
    X_scaled, _, _ = scale_features(X_clean)  # Modification ici
    
    ks = list(range(k_min, k_max + 1))
    inertias = []
    
    for k in ks:
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init='auto'
        )
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        logger.debug(f"k={k}: inertia={kmeans.inertia_:.2f}")
    
    return ks, inertias


def train_kmeans(
    X: pd.DataFrame,
    n_clusters: int,
    use_pca: bool = True,
    n_components: int = 2,
    random_state: int = 42
) -> Dict:
    """
    Entraîne un modèle KMeans sur les données.
    """
    logger.info(f"Entraînement KMeans avec {n_clusters} clusters")
    
    # Préparer les données
    X_clean = prepare_data_for_clustering(X)
    
    # Standardisation
    X_scaled, scaler, numeric_cols = scale_features(X_clean)  # Modification ici
    
    # PCA optionnel
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=random_state)
        X_transformed = pca.fit_transform(X_scaled)
        logger.info(f"PCA appliquée: {X_transformed.shape[1]} composantes, "
                   f"variance expliquée: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        pca = None
        X_transformed = X_scaled
    
    # KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto'
    )
    cluster_labels = kmeans.fit_predict(X_transformed)
    
    # Métriques de qualité
    silhouette = silhouette_score(X_transformed, cluster_labels)
    db_index = davies_bouldin_score(X_transformed, cluster_labels)
    
    result = {
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'cluster_labels': cluster_labels,
        'X_transformed': X_transformed,
        'X_clean': X_clean,
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_index,
        'inertia': kmeans.inertia_,
        'n_clusters': n_clusters,
        'feature_names': numeric_cols
    }
    
    logger.info(f"KMeans entraîné: silhouette={silhouette:.3f}, "
               f"DB index={db_index:.3f}, inertia={kmeans.inertia_:.2f}")
    
    return result


def prepare_data_for_clustering(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les données pour le clustering.
    """
    X_clean = X.copy()
    
    # 1. Supprimer les colonnes avec toutes les valeurs manquantes
    X_clean = X_clean.dropna(axis=1, how='all')
    
    # 2. Supprimer les colonnes non-numériques
    non_numeric_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"Suppression des colonnes non-numériques pour clustering: {non_numeric_cols}")
        X_clean = X_clean.drop(columns=non_numeric_cols)
    
    # 3. Remplir les valeurs manquantes
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].median())
    
    # 4. Vérifier qu'il reste des colonnes
    if X_clean.shape[1] == 0:
        raise ValueError("Aucune colonne numérique disponible pour clustering")
    
    logger.info(f"Données préparées pour clustering: {X_clean.shape}")
    return X_clean

def compute_silhouette_scores(
    X: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> Tuple[List[int], List[float]]:
    """
    Calcule les scores de silhouette pour différentes valeurs de k.
    
    Args:
        X: Données d'entrée
        k_min: Nombre minimum de clusters
        k_max: Nombre maximum de clusters
        random_state: Seed pour la reproductibilité
        
    Returns:
        Tuple (liste des k, liste des scores silhouette)
    """
    logger.info(f"Calcul des scores silhouette pour k de {k_min} à {k_max}")
    
    X_scaled, _ = scale_features(X)
    
    ks = list(range(k_min, k_max + 1))
    silhouette_scores = []
    
    for k in ks:
        if k == 1:
            # silhouette_score n'est pas défini pour k=1
            silhouette_scores.append(np.nan)
            continue
            
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init='auto'
        )
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        logger.debug(f"k={k}: silhouette={score:.3f}")
    
    return ks, silhouette_scores


def train_kmeans(
    X: pd.DataFrame,
    n_clusters: int,
    use_pca: bool = True,
    n_components: int = 2,
    random_state: int = 42
) -> Dict:
    """
    Entraîne un modèle KMeans sur les données.
    
    Args:
        X: Données d'entrée
        n_clusters: Nombre de clusters
        use_pca: Si True, applique PCA avant KMeans
        n_components: Nombre de composantes PCA si use_pca=True
        random_state: Seed pour la reproductibilité
        
    Returns:
        Dictionnaire avec le modèle et les résultats
    """
    logger.info(f"Entraînement KMeans avec {n_clusters} clusters")
    
    # Standardisation
    X_scaled, scaler = scale_features(X)
    
    # PCA optionnel
    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        X_transformed = pca.fit_transform(X_scaled)
        logger.info(f"PCA appliquée: {n_components} composantes, "
                   f"variance expliquée: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        pca = None
        X_transformed = X_scaled
    
    # KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto'
    )
    cluster_labels = kmeans.fit_predict(X_transformed)
    
    # Métriques de qualité
    silhouette = silhouette_score(X_transformed, cluster_labels)
    db_index = davies_bouldin_score(X_transformed, cluster_labels)
    
    result = {
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'cluster_labels': cluster_labels,
        'X_transformed': X_transformed,
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_index,
        'inertia': kmeans.inertia_,
        'n_clusters': n_clusters
    }
    
    logger.info(f"KMeans entraîné: silhouette={silhouette:.3f}, "
               f"DB index={db_index:.3f}, inertia={kmeans.inertia_:.2f}")
    
    return result


def get_cluster_profiles(
    X: pd.DataFrame,
    clustering_result: Dict
) -> pd.DataFrame:
    """
    Calcule les profils moyens de chaque cluster.
    
    Args:
        X: Données originales (non standardisées)
        clustering_result: Résultat de train_kmeans
        
    Returns:
        DataFrame avec statistiques par cluster
    """
    X_with_clusters = X.copy()
    X_with_clusters['cluster'] = clustering_result['cluster_labels']
    
    # Statistiques numériques
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        cluster_stats = X_with_clusters.groupby('cluster')[numeric_cols].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(3)
        
        # Aplatir les colonnes multi-index
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
        
        # Ajouter taille des clusters
        cluster_sizes = X_with_clusters['cluster'].value_counts().sort_index()
        cluster_stats['cluster_size'] = cluster_sizes.values
        cluster_stats['cluster_size_pct'] = (
            cluster_stats['cluster_size'] / len(X_with_clusters) * 100
        ).round(1)
        
        return cluster_stats
    
    return pd.DataFrame()


def suggest_optimal_k(
    X: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    method: str = 'elbow'
) -> int:
    """
    Suggère un nombre optimal de clusters basé sur différentes méthodes.
    
    Args:
        X: Données d'entrée
        k_min: Nombre minimum de clusters
        k_max: Nombre maximum de clusters
        method: Méthode de suggestion ('elbow', 'silhouette', 'combined')
        
    Returns:
        k suggéré
    """
    ks, inertias = compute_elbow_curve(X, k_min, k_max)
    
    if method == 'elbow':
        # Méthode du coude: trouver le point d'inflexion
        differences = np.diff(inertias)
        second_diff = np.diff(differences)
        
        if len(second_diff) > 0:
            # Trouver où la courbure change le plus
            optimal_idx = np.argmax(np.abs(second_diff)) + 1
            suggested_k = ks[optimal_idx]
        else:
            # Fallback: premier k où la diminution ralentit
            suggested_k = ks[2] if len(ks) > 2 else ks[0]
    
    elif method == 'silhouette':
        _, silhouette_scores = compute_silhouette_scores(X, k_min, k_max)
        # Ignorer k=1 (NaN)
        valid_scores = [(k, s) for k, s in zip(ks, silhouette_scores) if not np.isnan(s)]
        if valid_scores:
            suggested_k = max(valid_scores, key=lambda x: x[1])[0]
        else:
            suggested_k = ks[0]
    
    else:  # 'combined'
        # Moyenne des suggestions
        elbow_k = suggest_optimal_k(X, k_min, k_max, 'elbow')
        silhouette_k = suggest_optimal_k(X, k_min, k_max, 'silhouette')
        suggested_k = int(np.mean([elbow_k, silhouette_k]))
    
    logger.info(f"k suggéré ({method}): {suggested_k}")
    return suggested_k