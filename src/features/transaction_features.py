"""
Construction des features au niveau transaction.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from src.logging_config import get_logger
from src.data.preprocessor import preprocess_transactions, extract_datetime_features

logger = get_logger(__name__)


def build_transaction_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construit des features au niveau transaction.
    
    Args:
        df_raw: DataFrame des transactions brutes
        
    Returns:
        DataFrame des features transaction
    """
    logger.info("Construction des features transaction")
    
    # Prétraitement et extraction des features temporelles
    df = preprocess_transactions(df_raw)
    df = extract_datetime_features(df)
    
    # Features de base
    tx_features = pd.DataFrame(index=df.index)
    
    # Features numériques
    numeric_cols = ['product_amount', 'transaction_fee', 'cashback', 'loyalty_points']
    for col in numeric_cols:
        if col in df.columns:
            tx_features[col] = df[col].fillna(0)
    
    # Features temporelles
    if 'hour' in df.columns:
        tx_features['hour'] = df['hour']
        tx_features['is_night'] = df.get('is_night', 0)
    
    if 'day_of_week' in df.columns:
        tx_features['day_of_week'] = df['day_of_week']
        tx_features['is_weekend'] = df.get('is_weekend', 0)
    
    # Encodage one-hot des catégories principales
    categorical_cols = ['product_category', 'payment_method', 'device_type']
    
    for col in categorical_cols:
        if col in df.columns:
            # Garder seulement les catégories les plus fréquentes
            top_categories = df[col].value_counts().head(10).index.tolist()
            df[col + '_encoded'] = df[col].apply(
                lambda x: x if x in top_categories else 'Other'
            )
            
            # Encodage one-hot
            dummies = pd.get_dummies(df[col + '_encoded'], prefix=col, drop_first=True)
            tx_features = pd.concat([tx_features, dummies], axis=1)
    
    # Features agrégées par utilisateur (rollup)
    if 'user_id' in df.columns:
        user_stats = df.groupby('user_id').agg({
            'product_amount': ['mean', 'std'],
            'cashback': ['mean', 'std'],
        }).round(2)
        
        user_stats.columns = ['user_avg_amount', 'user_std_amount', 
                              'user_avg_cashback', 'user_std_cashback']
        
        # Joindre avec les transactions
        tx_features = tx_features.join(user_stats, on=df['user_id'])
        
        # Features comparatives
        tx_features['amount_vs_user_avg'] = (
            df['product_amount'] / tx_features['user_avg_amount'].replace(0, 1)
        ).replace([np.inf, -np.inf], 0).fillna(0)
        
        if 'cashback' in df.columns:
            tx_features['cashback_vs_user_avg'] = (
                df['cashback'] / tx_features['user_avg_cashback'].replace(0, 1)
            ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Traitement des valeurs manquantes
    numeric_cols = tx_features.select_dtypes(include=[np.number]).columns
    tx_features[numeric_cols] = tx_features[numeric_cols].fillna(0)
    
    logger.info(f"Features transaction construites: {tx_features.shape}")
    return tx_features


def get_transaction_feature_description() -> Dict[str, str]:
    """
    Retourne une description des features transaction.
    
    Returns:
        Dictionnaire {feature: description}
    """
    return {
        'product_amount': "Montant de la transaction",
        'transaction_fee': "Frais de transaction",
        'cashback': "Cashback reçu",
        'loyalty_points': "Points de fidélité gagnés",
        'hour': "Heure de la transaction (0-23)",
        'is_night': "Transaction nocturne (0-5h)",
        'day_of_week': "Jour de la semaine (0=lundi, 6=dimanche)",
        'is_weekend': "Transaction le week-end",
        'user_avg_amount': "Moyenne des montants de l'utilisateur",
        'user_std_amount': "Écart-type des montants de l'utilisateur",
        'amount_vs_user_avg': "Ratio montant / moyenne utilisateur",
        'user_avg_cashback': "Moyenne du cashback de l'utilisateur",
        'cashback_vs_user_avg': "Ratio cashback / moyenne utilisateur"
    }