"""
Construction des features au niveau transaction.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

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
    
    # 3. Encodage numérique des catégories
    categorical_cols = ['product_category', 'payment_method', 'device_type', 'location']
    
    for col in categorical_cols:
        if col in df.columns:
            # Encodage ordinal simple
            encoded_values, _ = pd.factorize(df[col])
            tx_features[f'{col}_encoded'] = encoded_values.astype(float)
    
    # 4. Features agrégées par utilisateur (rollup)
    if 'user_id' in df.columns and 'product_amount' in df.columns:
        try:
            # Ajouter d'abord user_id comme colonne
            if 'user_id' not in tx_features.columns:
                # Stocker user_id séparément pour les joins
                tx_features['user_id_temp'] = df['user_id'].astype(str)
            
            # Calculer les stats utilisateur
            user_stats = df.groupby('user_id').agg({
                'product_amount': ['mean', 'std'],
            }).round(2)
            
            # Aplatir les colonnes multi-index
            user_stats.columns = ['user_avg_amount', 'user_std_amount']
            user_stats = user_stats.reset_index()
            
            # Joindre les stats utilisateur
            if 'user_id_temp' in tx_features.columns:
                merged = pd.merge(
                    tx_features.reset_index(),
                    user_stats,
                    left_on='user_id_temp',
                    right_on='user_id',
                    how='left'
                ).set_index('index')
                
                # Remplacer tx_features par le merged version
                tx_features = merged.drop(columns=['user_id', 'user_id_temp'], errors='ignore')
            else:
                # Si pas de user_id_temp, utiliser l'index
                tx_features = tx_features.join(user_stats.set_index('user_id'), 
                                            on=df['user_id'])
            
            # Features comparatives (éviter division par zéro)
            if 'user_avg_amount' in tx_features.columns:
                tx_features['amount_vs_user_avg'] = (
                    df['product_amount'] / tx_features['user_avg_amount'].replace(0, 1)
                ).replace([np.inf, -np.inf], 0).fillna(0).astype(float)
                
        except Exception as e:
            logger.warning(f"Erreur dans les stats utilisateur: {e}")
    
    if 'user_id' in df.columns and 'cashback' in df.columns:
        try:
            # Calculer les stats cashback
            cashback_stats = df.groupby('user_id')['cashback'].agg(['mean', 'std']).round(2)
            cashback_stats.columns = ['user_avg_cashback', 'user_std_cashback']
            cashback_stats = cashback_stats.reset_index()
            
            # Stocker user_id temporairement pour le join
            if 'user_id' not in tx_features.columns:
                tx_features['user_id_temp'] = df['user_id'].astype(str)
            
            if 'user_id_temp' in tx_features.columns:
                merged = pd.merge(
                    tx_features.reset_index(),
                    cashback_stats,
                    left_on='user_id_temp',
                    right_on='user_id',
                    how='left'
                ).set_index('index')
                
                tx_features = merged.drop(columns=['user_id', 'user_id_temp'], errors='ignore')
            else:
                tx_features = tx_features.join(cashback_stats.set_index('user_id'), 
                                            on=df['user_id'])
            
            if 'user_avg_cashback' in tx_features.columns:
                tx_features['cashback_vs_user_avg'] = (
                    df['cashback'] / tx_features['user_avg_cashback'].replace(0, 1)
                ).replace([np.inf, -np.inf], 0).fillna(0).astype(float)
                
        except Exception as e:
            logger.warning(f"Erreur dans les stats cashback: {e}")
    
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
        logger.warning(f"Suppression des colonnes non-numériques: {non_numeric_cols}")
        tx_features = tx_features.drop(columns=non_numeric_cols)
    
    # 7. Remplir les NaN restants
    tx_features = tx_features.fillna(0)
    
    # 8. Vérifier qu'il y a des colonnes
    if tx_features.shape[1] == 0:
        raise ValueError("Aucune feature transaction numérique créée!")
    
    # 9. Supprimer les colonnes potentiellement dupliquées
    tx_features = tx_features.loc[:, ~tx_features.columns.duplicated()]
    
    logger.info(f"Features transaction construites: {tx_features.shape}")
    logger.info(f"Colonnes finales: {list(tx_features.columns)}")
    
    return tx_features


def build_transaction_features_simple(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Version simplifiée des features transaction (évite les joins problématiques).
    """
    logger.info("Construction simplifiée des features transaction")
    
    # Prétraitement
    df = preprocess_transactions(df_raw)
    df = extract_datetime_features(df)
    
    # Commencer avec un DataFrame vide
    tx_features = pd.DataFrame(index=df.index)
    
    # 1. Features numériques de base (sans dépendances)
    base_cols = ['product_amount', 'transaction_fee', 'cashback', 'loyalty_points']
    for col in base_cols:
        if col in df.columns:
            tx_features[col] = df[col].fillna(0).astype(float)
    
    # 2. Features temporelles
    time_features = []
    if 'hour' in df.columns:
        tx_features['hour'] = df['hour'].fillna(12).astype(float)
        time_features.append('hour')
    
    if 'day_of_week' in df.columns:
        tx_features['day_of_week'] = df['day_of_week'].fillna(0).astype(float)
        time_features.append('day_of_week')
    
    if 'is_weekend' in df.columns:
        tx_features['is_weekend'] = df['is_weekend'].fillna(0).astype(float)
        time_features.append('is_weekend')
    
    # 3. Encodage simple des catégories (sans one-hot)
    for col in ['product_category', 'payment_method', 'device_type']:
        if col in df.columns:
            # Simple factorize (plus stable que get_dummies)
            encoded, _ = pd.factorize(df[col])
            tx_features[f'{col}_code'] = encoded.astype(float)
    
    # 4. Features calculées simples (pas de joins)
    if 'product_amount' in tx_features.columns:
        # Statistiques globales (pas par utilisateur)
        global_mean = tx_features['product_amount'].mean()
        global_std = tx_features['product_amount'].std()
        
        tx_features['amount_zscore'] = (
            (tx_features['product_amount'] - global_mean) / global_std.replace(0, 1)
        ).fillna(0).astype(float)
        
        # Quartile
        tx_features['amount_quartile'] = pd.qcut(
            tx_features['product_amount'], 
            q=4, 
            labels=[0, 1, 2, 3],
            duplicates='drop'
        ).astype(float)
    
    if 'cashback' in tx_features.columns and 'product_amount' in tx_features.columns:
        tx_features['cashback_ratio'] = (
            tx_features['cashback'] / tx_features['product_amount'].replace(0, 1)
        ).replace([np.inf, -np.inf], 0).fillna(0).astype(float)
    
    # 5. Supprimer les NaN
    tx_features = tx_features.fillna(0)
    
    # 6. Supprimer les colonnes non-numériques
    non_numeric = tx_features.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        tx_features = tx_features.drop(columns=non_numeric)
    
    # 7. Supprimer les doublons de colonnes
    tx_features = tx_features.loc[:, ~tx_features.columns.duplicated()]
    
    logger.info(f"Features transaction simplifiées: {tx_features.shape}")
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
        'day_of_week': "Jour de la semaine (0=lundi, 6=dimanche)",
        'is_weekend': "Transaction le week-end (0=non, 1=oui)",
        'product_category_code': "Code de la catégorie produit",
        'payment_method_code': "Code de la méthode de paiement",
        'device_type_code': "Code du type d'appareil",
        'amount_zscore': "Score Z du montant (normalisation)",
        'amount_quartile': "Quartile du montant (0-3)",
        'cashback_ratio': "Ratio cashback/montant",
        'log_amount': "Logarithme du montant",
        'amount_vs_user_avg': "Ratio montant/moyenne utilisateur",
        'cashback_vs_user_avg': "Ratio cashback/moyenne utilisateur"
    }