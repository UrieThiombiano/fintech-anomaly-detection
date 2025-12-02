"""
Prétraitement des données brutes.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from datetime import datetime

from src.logging_config import get_logger
from src.utils import handle_missing_values

logger = get_logger(__name__)


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite les données de transactions brutes.
    
    Args:
        df: DataFrame des transactions brutes
        
    Returns:
        DataFrame prétraité
    """
    logger.info("Début du prétraitement des transactions")
    df_processed = df.copy()
    
    # Conversion des dates
    if 'transaction_date' in df_processed.columns:
        df_processed['transaction_date'] = pd.to_datetime(
            df_processed['transaction_date'], errors='coerce'
        )
        df_processed['transaction_hour'] = df_processed['transaction_date'].dt.hour
        df_processed['transaction_day'] = df_processed['transaction_date'].dt.day
        df_processed['transaction_month'] = df_processed['transaction_date'].dt.month
        df_processed['transaction_dayofweek'] = df_processed['transaction_date'].dt.dayofweek
    
    # Traitement des valeurs manquantes
    df_processed = handle_missing_values(df_processed, strategy='zero')
    
    # Encodage de variables catégorielles fréquentes
    categorical_cols = ['product_category', 'payment_method', 'device_type', 'location']
    for col in categorical_cols:
        if col in df_processed.columns:
            # Remplacement des valeurs rares par 'Other'
            value_counts = df_processed[col].value_counts()
            rare_values = value_counts[value_counts < 10].index
            if len(rare_values) > 0:
                df_processed[col] = df_processed[col].replace(rare_values, 'Other')
    
    logger.info("Prétraitement terminé")
    return df_processed


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des features temporelles à partir des dates.
    
    Args:
        df: DataFrame avec colonne transaction_date
        
    Returns:
        DataFrame avec features temporelles ajoutées
    """
    if 'transaction_date' not in df.columns:
        return df
    
    df_temp = df.copy()
    df_temp['transaction_date'] = pd.to_datetime(df_temp['transaction_date'])
    
    # Features temporelles
    df_temp['hour'] = df_temp['transaction_date'].dt.hour
    df_temp['day_of_week'] = df_temp['transaction_date'].dt.dayofweek
    df_temp['day_of_month'] = df_temp['transaction_date'].dt.day
    df_temp['month'] = df_temp['transaction_date'].dt.month
    df_temp['is_weekend'] = df_temp['day_of_week'].isin([5, 6]).astype(int)
    df_temp['is_night'] = ((df_temp['hour'] >= 0) & (df_temp['hour'] <= 5)).astype(int)
    
    return df_temp