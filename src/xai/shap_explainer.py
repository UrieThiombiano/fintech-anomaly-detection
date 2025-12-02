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
    """
    logger.info(f"Calcul SHAP sur échantillon de {sample_size} transactions")
    
    # Préparer les données
    try:
        # Assurer que tx_features est numérique
        tx_features_numeric = tx_features.select_dtypes(include=[np.number])
        
        if tx_features_numeric.empty:
            raise ValueError("Aucune colonne numérique dans tx_features")
        
        # Limiter la taille de l'échantillon
        sample_size = min(sample_size, len(tx_features_numeric))
        
        # Échantillonnage
        if sample_size < len(tx_features_numeric):
            rng = np.random.RandomState(random_state)
            indices = rng.choice(len(tx_features_numeric), size=sample_size, replace=False)
            X_sample = tx_features_numeric.iloc[indices].copy()
        else:
            indices = np.arange(len(tx_features_numeric))
            X_sample = tx_features_numeric.copy()
        
        # Standardisation
        X_scaled = scaler.transform(X_sample)
        
        # Explainer SHAP
        try:
            explainer = shap.TreeExplainer(iforest)
            shap_values = explainer.shap_values(X_scaled)
            
            # Pour Isolation Forest, shap_values est une liste avec un élément
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
        except Exception as e:
            logger.warning(f"SHAP TreeExplainer échoué, utilisation de KernelExplainer: {e}")
            # Fallback: KernelExplainer (plus lent mais plus robuste)
            background = shap.sample(X_scaled, 50)  # Échantillon de fond
            explainer = shap.KernelExplainer(iforest.predict, background)
            shap_values = explainer.shap_values(X_scaled, nsamples=100)
        
        result = {
            'X_sample': X_sample,
            'X_scaled': X_scaled,
            'shap_values': shap_values,
            'feature_names': tx_features_numeric.columns.tolist(),
            'indices': indices,
            'explainer': explainer,
            'sample_size': sample_size,
            'success': True
        }
        
        logger.info(f"SHAP calculé: shape={shap_values.shape}")
        
    except Exception as e:
        logger.error(f"Erreur dans compute_shap_for_isolation_forest: {e}")
        # Retourner un résultat vide en cas d'erreur
        result = {
            'X_sample': pd.DataFrame(),
            'X_scaled': np.array([]),
            'shap_values': np.array([]),
            'feature_names': [],
            'indices': np.array([]),
            'explainer': None,
            'sample_size': 0,
            'success': False,
            'error': str(e)
        }
    
    return result


def get_top_shap_features(
    shap_result: Dict,
    transaction_idx: int,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Récupère les features les plus importantes pour une transaction donnée.
    """
    if not shap_result.get('success', True):
        return pd.DataFrame({
            'feature': ['Erreur SHAP'],
            'shap_value': [0],
            'abs_shap': [0],
            'impact': ['Non disponible']
        })
    
    shap_values = shap_result['shap_values']
    feature_names = shap_result['feature_names']
    
    if len(shap_values) == 0 or len(feature_names) == 0:
        return pd.DataFrame({
            'feature': ['Données non disponibles'],
            'shap_value': [0],
            'abs_shap': [0],
            'impact': ['Non disponible']
        })
    
    if transaction_idx >= len(shap_values):
        logger.warning(f"Index {transaction_idx} hors limites, utilisation du premier")
        transaction_idx = 0
    
    # Récupérer les valeurs SHAP pour cette transaction
    shap_row = shap_values[transaction_idx]
    
    # Créer DataFrame
    df_shap = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_row,
        'abs_shap': np.abs(shap_row)
    })
    
    # Trier par importance
    top_n = min(top_n, len(df_shap))
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
    """
    if not shap_result.get('success', True):
        return {
            'global_importance': [],
            'n_transactions': 0,
            'n_features': 0,
            'shap_mean': 0,
            'shap_std': 0,
            'shap_min': 0,
            'shap_max': 0,
            'success': False
        }
    
    shap_values = shap_result['shap_values']
    feature_names = shap_result['feature_names']
    
    if len(shap_values) == 0 or len(feature_names) == 0:
        return {
            'global_importance': [],
            'n_transactions': 0,
            'n_features': 0,
            'shap_mean': 0,
            'shap_std': 0,
            'shap_min': 0,
            'shap_max': 0,
            'success': False
        }
    
    # Importance globale des features (moyenne des valeurs absolues SHAP)
    global_importance = np.mean(np.abs(shap_values), axis=0)
    
    top_n_features = min(top_n_features, len(feature_names))
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
        'shap_max': float(np.max(shap_values)),
        'success': True
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
    """
    if not shap_result.get('success', True):
        return "⚠️ Les explications SHAP ne sont pas disponibles pour le moment."
    
    try:
        # Récupérer les features importantes
        top_features = get_top_shap_features(shap_result, transaction_idx, top_n)
        
        if top_features.empty or 'Erreur' in top_features['feature'].iloc[0]:
            return "Aucune explication SHAP disponible pour cette transaction."
        
        # Récupérer les données originales de la transaction
        sample_idx = shap_result['indices'][transaction_idx] if transaction_idx < len(shap_result['indices']) else 0
        
        if sample_idx >= len(original_data):
            return "Transaction non trouvée dans les données originales."
        
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
            for _, row in positive_features.head(3).iterrows():  # Limiter à 3
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
            for _, row in negative_features.head(2).iterrows():  # Limiter à 2
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
            context_parts.append(f"montant de {amount:.2f}€")
        
        if 'cashback' in transaction_data:
            cashback = transaction_data['cashback']
            context_parts.append(f"cashback de {cashback:.2f}€")
        
        if 'product_category' in transaction_data:
            category = transaction_data['product_category']
            context_parts.append(f"catégorie '{category}'")
        
        if context_parts:
            explanation_parts.append(
                f"\n**Contexte :** Transaction avec {', '.join(context_parts)}."
            )
        
        return "\n".join(explanation_parts)
    
    except Exception as e:
        logger.error(f"Erreur dans explain_anomaly_in_french: {e}")
        return f"Erreur lors de la génération de l'explication: {str(e)}"