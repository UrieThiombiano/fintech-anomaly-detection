"""
Fonctions pour l'Analyse en Composantes Principales (PCA).
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.logging_config import get_logger
from src.utils import scale_features

logger = get_logger(__name__)


def compute_pca(
    X: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.9,
    random_state: int = 42
) -> Dict:
    """
    Calcule une PCA sur les données.
    """
    logger.info(f"Calcul de la PCA sur données de shape {X.shape}")
    
    # Vérifier et préparer les données
    X_clean = prepare_data_for_pca(X)
    
    # Standardisation - modification ici pour utiliser 3 retours
    X_scaled, scaler, numeric_cols = scale_features(X_clean)
    
    # Détermination du nombre de composantes si non spécifié
    if n_components is None:
        # PCA pour toutes les composantes possibles
        pca_temp = PCA(random_state=random_state)
        pca_temp.fit(X_scaled)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        
        # Trouver le nombre de composantes pour atteindre le seuil
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        logger.info(f"Nombre de composantes déterminé: {n_components} "
                   f"(variance cumulée: {cumsum[n_components-1]:.3f})")
    
    # PCA avec le nombre de composantes choisi
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    # Résultats
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Loadings (corrélations variables-composantes)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    result = {
        'scaler': scaler,
        'pca': pca,
        'X_pca': X_pca,
        'X_scaled': X_scaled,
        'explained_variance_ratio': explained_variance,
        'cumulative_variance': cumulative_variance,
        'loadings': loadings,
        'feature_names': numeric_cols if numeric_cols else list(range(X_scaled.shape[1])),
        'n_components': n_components,
        'original_data': X_clean  # Garder une copie des données originales nettoyées
    }
    
    logger.info(f"PCA terminée: {n_components} composantes, "
               f"variance totale expliquée: {cumulative_variance[-1]:.3f}")
    
    return result


def prepare_data_for_pca(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les données pour PCA en gérant les valeurs manquantes et non-numériques.
    """
    X_clean = X.copy()
    
    # 1. Supprimer les colonnes avec toutes les valeurs manquantes
    X_clean = X_clean.dropna(axis=1, how='all')
    
    # 2. Identifier les colonnes non-numériques
    non_numeric_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if non_numeric_cols:
        logger.info(f"Colonnes non-numériques trouvées: {non_numeric_cols}")
        
        # Essayer de convertir les colonnes catégorielles en numériques
        for col in non_numeric_cols:
            if X_clean[col].nunique() < 10:  # Peu de catégories -> one-hot encoding
                dummies = pd.get_dummies(X_clean[col], prefix=col, drop_first=True)
                X_clean = pd.concat([X_clean, dummies], axis=1)
            else:
                # Trop de catégories, essayer un encodage ordinal simple
                try:
                    X_clean[col] = pd.factorize(X_clean[col])[0]
                except:
                    pass
        
        # Supprimer les colonnes originales non-numériques
        X_clean = X_clean.drop(columns=non_numeric_cols)
    
    # 3. Remplir les valeurs manquantes
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].median())
    
    # 4. Vérifier qu'il reste des colonnes numériques
    if X_clean.shape[1] == 0:
        raise ValueError("Aucune colonne numérique disponible pour PCA après nettoyage")
    
    logger.info(f"Données préparées pour PCA: {X_clean.shape}")
    return X_clean

def get_pca_summary(pca_result: Dict) -> pd.DataFrame:
    """
    Crée un DataFrame récapitulatif de la PCA.
    
    Args:
        pca_result: Résultat de compute_pca
        
    Returns:
        DataFrame avec les informations principales
    """
    n_components = pca_result['n_components']
    
    summary = pd.DataFrame({
        'composante': [f'PC{i+1}' for i in range(n_components)],
        'variance_expliquee': pca_result['explained_variance_ratio'],
        'variance_cumulee': pca_result['cumulative_variance']
    })
    
    return summary


def get_top_loadings(pca_result: Dict, component: int = 0, n_features: int = 10) -> pd.DataFrame:
    """
    Récupère les variables les plus corrélées avec une composante donnée.
    
    Args:
        pca_result: Résultat de compute_pca
        component: Indice de la composante (0-based)
        n_features: Nombre de variables à retourner
        
    Returns:
        DataFrame avec variables et leurs loadings
    """
    if component >= pca_result['n_components']:
        raise ValueError(f"Component {component} n'existe pas. "
                        f"Nombre maximum: {pca_result['n_components'] - 1}")
    
    loadings = pca_result['loadings'][:, component]
    feature_names = pca_result['feature_names']
    
    df_loadings = pd.DataFrame({
        'variable': feature_names,
        'loading': loadings,
        'abs_loading': np.abs(loadings)
    })
    
    # Trier par valeur absolue
    df_loadings = df_loadings.sort_values('abs_loading', ascending=False).head(n_features)
    
    return df_loadings[['variable', 'loading']].reset_index(drop=True)


def suggest_optimal_components(pca_result: Dict, thresholds: List[float] = None) -> Dict:
    """
    Suggère des nombres optimaux de composantes basés sur différents seuils.
    
    Args:
        pca_result: Résultat de compute_pca (doit contenir toutes les composantes)
        thresholds: Liste des seuils de variance à considérer
        
    Returns:
        Dictionnaire avec suggestions
    """
    if thresholds is None:
        thresholds = [0.7, 0.8, 0.9, 0.95]
    
    cumulative_variance = pca_result['cumulative_variance']
    suggestions = {}
    
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        actual_variance = cumulative_variance[n_components - 1]
        suggestions[threshold] = {
            'n_components': n_components,
            'variance_expliquee': actual_variance
        }
    
    return suggestions