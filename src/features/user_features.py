"""
Construction des features au niveau utilisateur.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from src.logging_config import get_logger
from src.data.preprocessor import preprocess_transactions

logger = get_logger(__name__)


def build_user_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construit des features agrégées au niveau utilisateur.
    
    Args:
        df_raw: DataFrame des transactions brutes
        
    Returns:
        DataFrame des features utilisateur (index = user_id)
    """
    logger.info("Construction des features utilisateur")
    
    # Prétraitement
    df = preprocess_transactions(df_raw)
    
    # Vérification des colonnes nécessaires
    required_cols = ['user_id', 'product_amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes pour features utilisateur: {missing_cols}")
    
    # Agrégations de base
    user_features = df.groupby('user_id').agg({
        'transaction_id': 'count',
        'product_amount': ['sum', 'mean', 'std', 'min', 'max'],
    }).round(2)
    
    # Renommage des colonnes
    user_features.columns = [
        'nb_transactions',
        'total_amount', 'avg_amount', 'std_amount', 'min_amount', 'max_amount'
    ]
    
    # Cashback si disponible
    if 'cashback' in df.columns:
        cashback_features = df.groupby('user_id')['cashback'].agg([
            'sum', 'mean', 'std', 'max'
        ]).round(3)
        cashback_features.columns = [
            'total_cashback', 'avg_cashback', 'std_cashback', 'max_cashback'
        ]
        user_features = user_features.join(cashback_features)
        
        # Ratio cashback / montant
        user_features['cashback_ratio'] = (
            user_features['total_cashback'] / user_features['total_amount']
        ).replace([np.inf, -np.inf], 0).fillna(0).round(4)
    
    # Points de fidélité si disponible
    if 'loyalty_points' in df.columns:
        loyalty_features = df.groupby('user_id')['loyalty_points'].agg([
            'sum', 'mean', 'max'
        ]).round(2)
        loyalty_features.columns = [
            'total_loyalty_points', 'avg_loyalty_points', 'max_loyalty_points'
        ]
        user_features = user_features.join(loyalty_features)
    
    # Features catégorielles si disponibles
    categorical_features = []
    
    if 'product_category' in df.columns:
        # Nombre de catégories uniques
        user_features['nb_categories'] = df.groupby('user_id')['product_category'].nunique()
        
        # Catégorie préférée
        def get_top_category(series):
            if len(series) > 0:
                return series.value_counts().index[0]
            return None
        
        top_categories = df.groupby('user_id')['product_category'].apply(get_top_category)
        top_categories.name = 'top_category'
        user_features = user_features.join(top_categories)
    
    if 'payment_method' in df.columns:
        user_features['nb_payment_methods'] = df.groupby('user_id')['payment_method'].nunique()
    
    if 'device_type' in df.columns:
        user_features['nb_device_types'] = df.groupby('user_id')['device_type'].nunique()
    
    # Features temporelles
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Fréquence des transactions
        user_features['transaction_frequency'] = (
            user_features['nb_transactions'] / 
            (df.groupby('user_id')['transaction_date'].max() - 
             df.groupby('user_id')['transaction_date'].min()).dt.days.replace(0, 1)
        ).replace([np.inf, -np.inf], 0).fillna(0).round(4)
        
        # Dernière transaction
        last_transaction = df.groupby('user_id')['transaction_date'].max()
        user_features['days_since_last_transaction'] = (
            pd.Timestamp.now() - last_transaction
        ).dt.days
    
    # Traitement des valeurs manquantes
    numeric_cols = user_features.select_dtypes(include=[np.number]).columns
    user_features[numeric_cols] = user_features[numeric_cols].fillna(0)
    
    logger.info(f"Features utilisateur construites: {user_features.shape}")
    return user_features


def get_user_feature_description() -> Dict[str, str]:
    """
    Retourne une description des features utilisateur.
    
    Returns:
        Dictionnaire {feature: description}
    """
    return {
        'nb_transactions': "Nombre total de transactions",
        'total_amount': "Montant total dépensé",
        'avg_amount': "Montant moyen par transaction",
        'std_amount': "Écart-type des montants",
        'min_amount': "Montant minimum",
        'max_amount': "Montant maximum",
        'total_cashback': "Cashback total reçu",
        'avg_cashback': "Cashback moyen par transaction",
        'cashback_ratio': "Ratio cashback/montant total",
        'total_loyalty_points': "Points de fidélité totaux",
        'avg_loyalty_points': "Points de fidélité moyens",
        'nb_categories': "Nombre de catégories de produits différentes",
        'nb_payment_methods': "Nombre de méthodes de paiement utilisées",
        'nb_device_types': "Nombre de types d'appareils utilisés",
        'transaction_frequency': "Fréquence moyenne des transactions (par jour)",
        'days_since_last_transaction': "Jours depuis la dernière transaction"
    }