"""
Explications SHAP pour les modèles de détection d'anomalies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import shap

from src.logging_config import get_logger

logger = get_logger(__name__)


def compute_shap_for_isolation_forest(
    iforest,
    scaler,
    tx_features: pd.DataFrame,
    sample_size: int = 200,
    random_state: int = 42
) -> Dict:
    """
    Calcule les valeurs SHAP pour un modèle Isolation Forest.
    
    Args:
        iforest: Modèle Isolation Forest entraîné
        scaler: StandardScaler utilisé pour l'entraînement
        tx_features: Features transaction (non standardisées)
        sample_size: Taille de l'échantillon pour SHAP
        random_state: Seed pour la reproductibilité
        
    Returns:
        Dictionnaire avec résultats SHAP
    """
    logger.info(f"Calcul SHAP sur échantillon de {sample_size} transactions")
    
    # Échantillonnage
    if sample_size < len(tx_features):
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(tx_features), size=sample_size, replace=False)
        X_sample = tx_features.iloc[indices].copy()
    else:
        indices = np.arange(len(tx_features))
        X_sample = tx_features.copy()
    
    # Standardisation
    X_scaled = scaler.transform(X_sample)
    
    # Explainer SHAP
    explainer = shap.TreeExplainer(iforest)
    
    # Calcul des valeurs SHAP
    shap_values = explainer.shap_values(X_scaled)
    
    # Pour Isolation Forest, shap_values est une liste avec un élément
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Les valeurs SHAP sont les contributions au score d'anomalie
    # Valeurs positives augmentent le score (plus anormal)
    
    result = {
        'X_sample': X_sample,
        'X_scaled': X_scaled,
        'shap_values': shap_values,
        'feature_names': tx_features.columns.tolist(),
        'indices': indices,
        'explainer': explainer,
        'sample_size': sample_size
    }
    
    logger.info(f"SHAP calculé: shape des valeurs = {shap_values.shape}")
    return result


def get_top_shap_features(
    shap_result: Dict,
    transaction_idx: int,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Récupère les features les plus importantes pour une transaction donnée.
    
    Args:
        shap_result: Résultat de compute_shap_for_isolation_forest
        transaction_idx: Index de la transaction dans l'échantillon
        top_n: Nombre de features à retourner
        
    Returns:
        DataFrame avec features et valeurs SHAP
    """
    shap_values = shap_result['shap_values']
    feature_names = shap_result['feature_names']
    
    if transaction_idx >= len(shap_values):
        raise ValueError(f"Index {transaction_idx} hors limites "
                        f"(max: {len(shap_values)-1})")
    
    # Récupérer les valeurs SHAP pour cette transaction
    shap_row = shap_values[transaction_idx]
    
    # Créer DataFrame
    df_shap = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_row,
        'abs_shap': np.abs(shap_row)
    })
    
    # Trier par importance
    df_shap = df_shap.sort_values('abs_shap', ascending=False).head(top_n)
    
    # Ajouter l'impact (positif/négatif)
    df_shap['impact'] = df_shap['shap_value'].apply(
        lambda x: 'Augmente anomalie' if x > 0 else 'Réduit anomalie'
    )
    
    return df_shap[['feature', 'shap_value', 'abs_shap', 'impact']].reset_index(drop=True)


def generate_shap_summary(
    shap_result: Dict,
    top_n_features: int = 15
) -> Dict:
    """
    Génère un résumé des explications SHAP.
    
    Args:
        shap_result: Résultat de compute_shap_for_isolation_forest
        top_n_features: Nombre de features globales à inclure
        
    Returns:
        Dictionnaire avec résumé
    """
    shap_values = shap_result['shap_values']
    feature_names = shap_result['feature_names']
    
    # Importance globale des features (moyenne des valeurs absolues SHAP)
    global_importance = np.mean(np.abs(shap_values), axis=0)
    
    df_global = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': global_importance
    }).sort_values('mean_abs_shap', ascending=False).head(top_n_features)
    
    # Statistiques
    summary = {
        'global_importance': df_global.to_dict('records'),
        'n_transactions': len(shap_values),
        'n_features': len(feature_names),
        'shap_mean': float(np.mean(shap_values)),
        'shap_std': float(np.std(shap_values)),
        'shap_min': float(np.min(shap_values)),
        'shap_max': float(np.max(shap_values))
    }
    
    return summary


def explain_anomaly_in_french(
    shap_result: Dict,
    transaction_idx: int,
    original_data: pd.DataFrame,
    top_n: int = 5
) -> str:
    """
    Génère une explication en français pour une anomalie.
    
    Args:
        shap_result: Résultat de compute_shap_for_isolation_forest
        transaction_idx: Index de la transaction dans l'échantillon
        original_data: DataFrame original avec les données brutes
        top_n: Nombre de features à inclure dans l'explication
        
    Returns:
        Explication en français
    """
    # Récupérer les features importantes
    top_features = get_top_shap_features(shap_result, transaction_idx, top_n)
    
    # Récupérer les données originales de la transaction
    sample_idx = shap_result['indices'][transaction_idx]
    transaction_data = original_data.iloc[sample_idx]
    
    # Construction de l'explication
    explanation_parts = []
    
    # Introduction
    explanation_parts.append(
        f"La transaction {transaction_data.get('transaction_id', 'N/A')} "
        f"de l'utilisateur {transaction_data.get('user_id', 'N/A')} a été "
        f"détectée comme anormale principalement en raison des caractéristiques suivantes :"
    )
    
    # Features augmentant l'anomalie
    positive_features = top_features[top_features['shap_value'] > 0]
    if not positive_features.empty:
        pos_list = []
        for _, row in positive_features.iterrows():
            feature = row['feature']
            shap_val = row['shap_value']
            
            # Récupérer la valeur de la feature si disponible
            if feature in transaction_data:
                value = transaction_data[feature]
                pos_list.append(f"{feature} (valeur: {value:.2f}, contribution: +{shap_val:.3f})")
            else:
                pos_list.append(f"{feature} (contribution: +{shap_val:.3f})")
        
        if pos_list:
            explanation_parts.append(
                "\n**Facteurs augmentant l'anomalie :**\n- " + "\n- ".join(pos_list)
            )
    
    # Features réduisant l'anomalie
    negative_features = top_features[top_features['shap_value'] < 0]
    if not negative_features.empty:
        neg_list = []
        for _, row in negative_features.iterrows():
            feature = row['feature']
            shap_val = row['shap_value']
            
            if feature in transaction_data:
                value = transaction_data[feature]
                neg_list.append(f"{feature} (valeur: {value:.2f}, contribution: {shap_val:.3f})")
            else:
                neg_list.append(f"{feature} (contribution: {shap_val:.3f})")
        
        if neg_list:
            explanation_parts.append(
                "\n**Facteurs réduisant l'anomalie :**\n- " + "\n- ".join(neg_list)
            )
    
    # Contexte supplémentaire
    context_parts = []
    if 'product_amount' in transaction_data:
        amount = transaction_data['product_amount']
        context_parts.append(f"montant de {amount:.2f}")
    
    if 'cashback' in transaction_data:
        cashback = transaction_data['cashback']
        context_parts.append(f"cashback de {cashback:.2f}")
    
    if 'product_category' in transaction_data:
        category = transaction_data['product_category']
        context_parts.append(f"catégorie '{category}'")
    
    if context_parts:
        explanation_parts.append(
            f"\n**Contexte :** Transaction avec {', '.join(context_parts)}."
        )
    
    return "\n".join(explanation_parts)