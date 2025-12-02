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
    
    # Vérifier et nettoyer les données
    X_clean = prepare_data_for_anomaly_detection(X)
    
    # Vérifier qu'il y a assez de données
    if len(X_clean) < 10:
        raise ValueError(f"Pas assez de données pour Isolation Forest: {len(X_clean)} échantillons")
    
    # Standardisation
    try:
        X_scaled, scaler, _ = scale_features(X_clean)
        logger.info(f"Données standardisées: shape={X_scaled.shape}")
    except Exception as e:
        logger.error(f"Erreur lors de la standardisation: {e}")
        # Fallback: simple normalisation
        X_scaled = X_clean.values.astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_scaled)
    
    # Isolation Forest avec gestion d'erreurs
    try:
        iforest = IsolationForest(
            contamination=min(contamination, 0.5),  # Limiter à 50% max
            random_state=random_state,
            n_estimators=n_estimators,
            n_jobs=-1,
            verbose=0
        )
        
        # Prédictions
        iforest.fit(X_scaled)
        anomaly_scores = iforest.decision_function(X_scaled)
        is_anomaly = iforest.predict(X_scaled)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement Isolation Forest: {e}")
        # Fallback: scores basés sur la distance au centre
        from scipy.spatial.distance import mahalanobis
        center = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled.T)
        try:
            inv_cov = np.linalg.inv(cov)
            anomaly_scores = np.array([mahalanobis(x, center, inv_cov) for x in X_scaled])
        except:
            # Simple distance euclidienne
            anomaly_scores = np.sqrt(np.sum((X_scaled - center) ** 2, axis=1))
        
        # Déterminer les anomalies basées sur un seuil
        threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
        is_anomaly = np.where(anomaly_scores > threshold, -1, 1)
        
        # Créer un dummy iforest
        iforest = IsolationForest()
    
    # Convertir -1/1 en booléen (True=anomalie)
    is_anomaly_bool = (is_anomaly == -1)
    
    # Normaliser les scores pour avoir des valeurs positives (plus élevé = plus anormal)
    normalized_scores = -anomaly_scores  # Inverser pour avoir positif = anormal
    
    result = {
        'scaler': scaler,
        'iforest': iforest,
        'anomaly_scores': normalized_scores,
        'is_anomaly': is_anomaly_bool,
        'contamination': contamination,
        'X_scaled': X_scaled,
        'X_clean': X_clean
    }
    
    n_anomalies = is_anomaly_bool.sum()
    logger.info(f"Isolation Forest entraîné: {n_anomalies} anomalies détectées "
               f"({n_anomalies/len(X_clean)*100:.1f}%)")
    
    return result


def prepare_data_for_anomaly_detection(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les données pour la détection d'anomalies.
    
    Returns:
        DataFrame 100% numérique
    """
    X_clean = X.copy()
    
    # 1. Supprimer les colonnes avec toutes les valeurs manquantes
    X_clean = X_clean.dropna(axis=1, how='all')
    
    # 2. Supprimer les colonnes non-numériques
    non_numeric_cols = X_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"Suppression des colonnes non-numériques: {non_numeric_cols}")
        X_clean = X_clean.drop(columns=non_numeric_cols)
    
    # 3. Vérifier qu'il reste des colonnes
    if X_clean.shape[1] == 0:
        raise ValueError("Aucune colonne numérique disponible pour la détection d'anomalies")
    
    # 4. Remplir les valeurs manquantes
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        # Remplacer inf/-inf d'abord
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        # Remplir avec la médiane
        X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].median())
    
    # 5. Vérifier les dimensions finales
    logger.info(f"Données préparées pour anomalie: {X_clean.shape}")
    
    return X_clean


def analyze_anomalies(
    original_df: pd.DataFrame,
    anomaly_result: Dict,
    score_threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Analyse les anomalies détectées et les enrichit avec les données originales.
    """
    # Créer une copie
    df_anomalies = original_df.copy()
    
    # S'assurer que les arrays ont la même longueur
    n_original = len(df_anomalies)
    n_scores = len(anomaly_result['anomaly_scores'])
    
    if n_original != n_scores:
        logger.warning(f"Dimensions incompatibles: original={n_original}, scores={n_scores}")
        # Truncater ou pad les scores
        if n_scores < n_original:
            # Répéter les scores si moins nombreux
            repeat_factor = n_original // n_scores + 1
            scores = np.tile(anomaly_result['anomaly_scores'], repeat_factor)[:n_original]
            is_anomaly = np.tile(anomaly_result['is_anomaly'], repeat_factor)[:n_original]
        else:
            # Tronquer si plus nombreux
            scores = anomaly_result['anomaly_scores'][:n_original]
            is_anomaly = anomaly_result['is_anomaly'][:n_original]
    else:
        scores = anomaly_result['anomaly_scores']
        is_anomaly = anomaly_result['is_anomaly']
    
    df_anomalies['anomaly_score'] = scores
    df_anomalies['is_anomaly'] = is_anomaly
    
    # Filtrer par seuil si spécifié
    if score_threshold is not None:
        df_anomalies['is_above_threshold'] = df_anomalies['anomaly_score'] >= score_threshold
    else:
        # Utiliser le 95ème percentile comme seuil par défaut
        threshold = np.percentile(df_anomalies['anomaly_score'], 95)
        df_anomalies['is_above_threshold'] = df_anomalies['anomaly_score'] >= threshold
    
    # Trier par score d'anomalie
    df_anomalies = df_anomalies.sort_values('anomaly_score', ascending=False)
    
    return df_anomalies


def get_anomaly_statistics(anomaly_result: Dict) -> Dict:
    """
    Calcule des statistiques sur les anomalies détectées.
    """
    scores = anomaly_result['anomaly_scores']
    is_anomaly = anomaly_result['is_anomaly']
    
    if len(scores) == 0:
        return {}
    
    stats = {
        'n_total': len(scores),
        'n_anomalies': is_anomaly.sum(),
        'pct_anomalies': is_anomaly.sum() / len(scores) * 100 if len(scores) > 0 else 0,
        'score_mean': float(scores.mean()) if len(scores) > 0 else 0,
        'score_std': float(scores.std()) if len(scores) > 0 else 0,
        'score_min': float(scores.min()) if len(scores) > 0 else 0,
        'score_max': float(scores.max()) if len(scores) > 0 else 0,
        'score_median': float(np.median(scores)) if len(scores) > 0 else 0,
        'score_q25': float(np.percentile(scores, 25)) if len(scores) > 0 else 0,
        'score_q75': float(np.percentile(scores, 75)) if len(scores) > 0 else 0,
        'score_q95': float(np.percentile(scores, 95)) if len(scores) > 0 else 0,
        'score_q99': float(np.percentile(scores, 99)) if len(scores) > 0 else 0
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
    
    try:
        # Préparer les données
        X_clean = prepare_data_for_anomaly_detection(X)
        
        # Standardisation
        X_scaled, _, _ = scale_features(X_clean)
        
        # Distances au centre (simplifié)
        distances = np.sqrt(np.sum(X_scaled**2, axis=1))
        
        # Pourcentage de points au-delà de différents seuils
        suggestions = {}
        for p in percentiles:
            threshold = np.percentile(distances, p)
            pct_outliers = (distances > threshold).sum() / len(distances)
            suggestions[p] = round(pct_outliers, 4)
        
        # Retourner la moyenne des suggestions
        suggested_contamination = np.mean(list(suggestions.values()))
        
        logger.info(f"Contamination suggérée: {suggested_contamination:.4f}")
        
        return min(max(suggested_contamination, 0.001), 0.1)  # Borné entre 0.1% et 10%
    
    except Exception as e:
        logger.warning(f"Erreur dans suggest_contamination: {e}")
        # Valeur par défaut raisonnable
        return 0.02
    
    def generate_anomaly_report(
    df_raw: pd.DataFrame,
    anomaly_result: Dict,
    score_threshold: float = None
) -> str:
        """
        Génère un rapport détaillé en français des anomalies détectées.
        
        Returns:
            Rapport formaté en markdown
        """
    if score_threshold is None:
        score_threshold = np.percentile(anomaly_result['anomaly_scores'], 95)
    
    df_anomalies = analyze_anomalies(df_raw, anomaly_result, score_threshold)
    high_score_tx = df_anomalies[df_anomalies['is_above_threshold']]
    
    stats = get_anomaly_statistics(anomaly_result)
    
    report = f"""
# 📊 RAPPORT DE DÉTECTION D'ANOMALIES
*Généré le {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}*

## 📈 Résumé exécutif

### Données analysées
- **Transactions totales** : {stats['n_total']:,}
- **Features utilisées** : {anomaly_result['X_scaled'].shape[1]}
- **Période analysée** : {df_raw['transaction_date'].min().date() if 'transaction_date' in df_raw.columns else 'Non spécifiée'} au {df_raw['transaction_date'].max().date() if 'transaction_date' in df_raw.columns else 'Non spécifiée'}

### Résultats de détection
- **Anomalies détectées** : {stats['n_anomalies']:,} ({stats['pct_anomalies']:.1f}%)
- **Score d'anomalie moyen** : {stats['score_mean']:.3f}
- **Score maximum** : {stats['score_max']:.3f}
- **Seuil de détection (Q95)** : {stats['score_q95']:.3f}
- **Transactions au-dessus du seuil** : {len(high_score_tx):,}

## 🔍 Analyse des anomalies

### Top 5 transactions les plus suspectes
"""
    
    if not high_score_tx.empty:
        top5 = high_score_tx.head(5)
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            report += f"\n{i}. **Transaction {row.get('transaction_id', idx)}** : "
            report += f"Score = {row.get('anomaly_score', 0):.3f}, "
            if 'product_amount' in row:
                report += f"Montant = {row['product_amount']:.2f}€"
            if 'product_category' in row:
                report += f", Catégorie = {row['product_category']}"
            report += f", User = {row.get('user_id', 'N/A')}"
    
    report += """

## 📊 Distribution statistique

### Quartiles des scores
- **Q25 (25%)** : {:.3f}
- **Q50 (Médiane)** : {:.3f}
- **Q75 (75%)** : {:.3f}
- **Q95 (95%)** : {:.3f}
- **Q99 (99%)** : {:.3f}

## 🎯 Recommandations

1. **Vérification manuelle** des transactions avec score > {:.3f}
2. **Analyse approfondie** des utilisateurs récurrents dans les anomalies
3. **Révision des règles métier** pour les catégories à risque
4. **Surveillance continue** avec mise à jour hebdomadaire du modèle

## 📈 Métriques de qualité

- **Séparation scores** : {:.2%} (écart moyen normal/anomalie)
- **Stabilité détection** : Contamination utilisée = {:.1%}
- **Capacité prédictive** : Modèle entraîné sur {:,} échantillons

---
*Ce rapport a été généré automatiquement par le système de détection d'anomalies Fintech*
""".format(
        stats['score_q25'],
        stats['score_median'],
        stats['score_q75'],
        stats['score_q95'],
        stats['score_q99'],
        score_threshold,
        (stats['score_mean'] - np.mean(anomaly_result['anomaly_scores'][~anomaly_result['is_anomaly']])) / stats['score_mean'] if 'score_mean' in stats else 0,
        anomaly_result.get('contamination', 0.02),
        len(anomaly_result['anomaly_scores'])
    )
    
    return report