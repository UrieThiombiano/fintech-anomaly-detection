"""
Fonctions pour la détection d'anomalies (Isolation Forest).
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.logging_config import get_logger
from src.utils import scale_features

logger = get_logger(__name__)


def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42,
    n_estimators: int = 100
) -> Dict:
    """
    Entraîne un modèle Isolation Forest pour la détection d'anomalies.
    """
    logger.info(f"Entraînement Isolation Forest avec contamination={contamination}")
    
    # CORRECTION ICI : 3 valeurs au lieu de 2
    X_scaled, scaler, _ = scale_features(X)
    
    # Isolation Forest
    iforest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=-1
    )
    
    # Prédictions
    anomaly_scores = iforest.decision_function(X_scaled)
    is_anomaly = iforest.predict(X_scaled)
    
    # Convertir -1/1 en booléen (True=anomalie)
    is_anomaly_bool = (is_anomaly == -1)
    
    # Normaliser les scores pour avoir des valeurs positives (plus élevé = plus anormal)
    # Les scores de decision_function sont négatifs pour les anomalies
    normalized_scores = -anomaly_scores  # Inverser pour avoir positif = anormal
    
    result = {
        'scaler': scaler,
        'iforest': iforest,
        'anomaly_scores': normalized_scores,
        'is_anomaly': is_anomaly_bool,
        'contamination': contamination,
        'X_scaled': X_scaled
    }
    
    n_anomalies = is_anomaly_bool.sum()
    logger.info(f"Isolation Forest entraîné: {n_anomalies} anomalies détectées "
               f"({n_anomalies/len(X)*100:.1f}%)")
    
    return result


def analyze_anomalies(
    original_df: pd.DataFrame,
    anomaly_result: Dict,
    score_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Analyse les anomalies détectées et les enrichit avec les données originales.
    """
    df_anomalies = original_df.copy()
    df_anomalies['anomaly_score'] = anomaly_result['anomaly_scores']
    df_anomalies['is_anomaly'] = anomaly_result['is_anomaly']
    
    # Filtrer par seuil si spécifié
    if score_threshold is not None:
        df_anomalies['is_above_threshold'] = df_anomalies['anomaly_score'] >= score_threshold
    else:
        df_anomalies['is_above_threshold'] = df_anomalies['is_anomaly']
    
    # Trier par score d'anomalie
    df_anomalies = df_anomalies.sort_values('anomaly_score', ascending=False)
    
    return df_anomalies


def get_anomaly_statistics(anomaly_result: Dict) -> Dict:
    """
    Calcule des statistiques sur les anomalies détectées.
    """
    scores = anomaly_result['anomaly_scores']
    is_anomaly = anomaly_result['is_anomaly']
    
    stats = {
        'n_total': len(scores),
        'n_anomalies': is_anomaly.sum(),
        'pct_anomalies': is_anomaly.sum() / len(scores) * 100,
        'score_mean': scores.mean(),
        'score_std': scores.std(),
        'score_min': scores.min(),
        'score_max': scores.max(),
        'score_median': np.median(scores),
        'score_q25': np.percentile(scores, 25),
        'score_q75': np.percentile(scores, 75),
        'score_q95': np.percentile(scores, 95),
        'score_q99': np.percentile(scores, 99)
    }
    
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}


def suggest_contamination(
    X: pd.DataFrame,
    percentiles: List[float] = None
) -> float:
    """
    Suggère une valeur de contamination basée sur les outliers statistiques.
    """
    if percentiles is None:
        percentiles = [95, 97.5, 99, 99.5]
    
    # Calculer les distances de Mahalanobis comme proxy pour les outliers
    from scipy.stats import chi2
    
    # CORRECTION ICI : 3 valeurs au lieu de 2
    X_scaled, _, _ = scale_features(X)
    
    # Distances au centre (simplifié)
    distances = np.sqrt(np.sum(X_scaled**2, axis=1))
    
    # Pourcentage de points au-delà de différents seuils
    suggestions = {}
    for p in percentiles:
        # Seuil théorique pour distribution normale
        threshold = np.percentile(distances, p)
        pct_outliers = (distances > threshold).sum() / len(distances)
        suggestions[p] = round(pct_outliers, 4)
    
    # Retourner la moyenne des suggestions
    suggested_contamination = np.mean(list(suggestions.values()))
    
    logger.info(f"Contamination suggérée: {suggested_contamination:.4f} "
               f"(basé sur {percentiles})")
    
    return min(max(suggested_contamination, 0.001), 0.1)  # Borné entre 0.1% et 10%