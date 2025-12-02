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
    Construit des features au niveau transaction (100% numériques).
    
    Args:
        df_raw: DataFrame des transactions brutes
        
    Returns:
        DataFrame des features transaction (toutes numériques)
    """
    logger.info("Construction des features transaction")
    
    # Prétraitement et extraction des features temporelles
    df = preprocess_transactions(df_raw)
    df = extract_datetime_features(df)
    
    # Features de base (uniquement numériques)
    tx_features = pd.DataFrame(index=df.index)
    
    # 1. Features numériques directes
    numeric_cols = ['product_amount', 'transaction_fee', 'cashback', 'loyalty_points']
    for col in numeric_cols:
        if col in df.columns:
            tx_features[col] = df[col].fillna(0).astype(float)
    
    # 2. Features temporelles (déjà numériques)
    time_cols = ['hour', 'is_night', 'day_of_week', 'is_weekend', 
                'day_of_month', 'month', 'transaction_hour']
    for col in time_cols:
        if col in df.columns:
            tx_features[col] = df[col].fillna(0).astype(float)
    
    # 3. Encodage numérique des catégories (pas de one-hot)
    categorical_cols = ['product_category', 'payment_method', 'device_type', 'location']
    
    for col in categorical_cols:
        if col in df.columns:
            # Encodage ordinal simple (pas one-hot pour éviter trop de dimensions)
            tx_features[f'{col}_encoded'] = pd.factorize(df[col])[0].astype(float)
    
    # 4. Features agrégées par utilisateur (rollup)
    if 'user_id' in df.columns and 'product_amount' in df.columns:
        # Éviter les erreurs de groupe vide
        valid_users = df['user_id'].dropna()
        if len(valid_users) > 0:
            user_stats = df.groupby('user_id').agg({
                'product_amount': ['mean', 'std'],
            }).round(2)
            
            # Gérer les noms de colonnes multi-index
            user_stats.columns = ['user_avg_amount', 'user_std_amount']
            
            # Joindre avec les transactions
            tx_features = tx_features.join(user_stats, on=df['user_id'])
            
            # Features comparatives (éviter division par zéro)
            tx_features['amount_vs_user_avg'] = (
                df['product_amount'] / tx_features['user_avg_amount'].replace(0, 1)
            ).replace([np.inf, -np.inf], 0).fillna(0).astype(float)
        
        if 'cashback' in df.columns:
            cashback_stats = df.groupby('user_id')['cashback'].agg(['mean', 'std']).round(2)
            cashback_stats.columns = ['user_avg_cashback', 'user_std_cashback']
            
            tx_features = tx_features.join(cashback_stats, on=df['user_id'])
            
            tx_features['cashback_vs_user_avg'] = (
                df['cashback'] / tx_features['user_avg_cashback'].replace(0, 1)
            ).replace([np.inf, -np.inf], 0).fillna(0).astype(float)
    
    # 5. Features supplémentaires
    if 'product_amount' in tx_features.columns:
        # Log du montant (pour gérer la skewness)
        tx_features['log_amount'] = np.log1p(tx_features['product_amount']).astype(float)
        
        # Interaction features
        if 'cashback' in tx_features.columns:
            tx_features['amount_cashback_ratio'] = (
                tx_features['cashback'] / tx_features['product_amount'].replace(0, 1)
            ).replace([np.inf, -np.inf], 0).fillna(0).astype(float)
    
    # 6. Assurer que toutes les colonnes sont numériques
    non_numeric_cols = tx_features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"Suppression des colonnes non-numériques dans tx_features: {non_numeric_cols}")
        tx_features = tx_features.drop(columns=non_numeric_cols)
    
    # 7. Remplir les NaN restants
    tx_features = tx_features.fillna(0)
    
    # 8. Vérifier qu'il y a des colonnes
    if tx_features.shape[1] == 0:
        raise ValueError("Aucune feature transaction numérique créée!")
    
    logger.info(f"Features transaction construites: {tx_features.shape}")
    logger.info(f"Types de données: {tx_features.dtypes.unique()}")
    
    return tx_features


def get_transaction_feature_description() -> Dict[str, str]:
    """
    Retourne une description des features transaction.
    
    Returns:
        Dictionnaire {feature: description}
    """
    return {
        'product_amount': "Montant de la transaction (€)",
        'transaction_fee': "Frais de transaction (€)",
        'cashback': "Cashback reçu (€)",
        'loyalty_points': "Points de fidélité gagnés",
        'hour': "Heure de la transaction (0-23)",
        'is_night': "Transaction nocturne (0=non, 1=oui)",
        'day_of_week': "Jour de la semaine (0=lundi, 6=dimanche)",
        'is_weekend': "Transaction le week-end (0=non, 1=oui)",
        'product_category_encoded': "Catégorie encodée (numérique)",
        'payment_method_encoded': "Méthode de paiement encodée (numérique)",
        'device_type_encoded': "Type d'appareil encodé (numérique)",
        'location_encoded': "Localisation encodée (numérique)",
        'user_avg_amount': "Moyenne des montants de l'utilisateur (€)",
        'user_std_amount': "Écart-type des montants de l'utilisateur (€)",
        'amount_vs_user_avg': "Ratio montant / moyenne utilisateur",
        'user_avg_cashback': "Moyenne du cashback de l'utilisateur (€)",
        'cashback_vs_user_avg': "Ratio cashback / moyenne utilisateur",
        'log_amount': "Logarithme du montant (pour normalisation)",
        'amount_cashback_ratio': "Ratio cashback/montant"
    }