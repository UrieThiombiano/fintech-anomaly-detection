"""
Construction des features au niveau utilisateur.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

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
        
        # Fréquence de la catégorie préférée
        def get_top_category_freq(series):
            if len(series) > 0:
                top_cat = series.value_counts().index[0]
                return (series == top_cat).sum() / len(series)
            return 0
        
        top_cat_freq = df.groupby('user_id')['product_category'].apply(get_top_category_freq)
        top_cat_freq.name = 'top_category_freq'
        user_features = user_features.join(top_cat_freq)
    
    if 'payment_method' in df.columns:
        user_features['nb_payment_methods'] = df.groupby('user_id')['payment_method'].nunique()
        
        # Méthode de paiement préférée
        def get_top_payment(series):
            if len(series) > 0:
                return series.value_counts().index[0]
            return None
        
        top_payment = df.groupby('user_id')['payment_method'].apply(get_top_payment)
        top_payment.name = 'top_payment_method'
        user_features = user_features.join(top_payment)
    
    if 'device_type' in df.columns:
        user_features['nb_device_types'] = df.groupby('user_id')['device_type'].nunique()
    
    if 'location' in df.columns:
        user_features['nb_locations'] = df.groupby('user_id')['location'].nunique()
    
    # Features temporelles
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Fréquence des transactions
        date_range = df.groupby('user_id')['transaction_date'].agg(['min', 'max'])
        date_range['range_days'] = (date_range['max'] - date_range['min']).dt.days
        
        # Éviter la division par zéro
        date_range['range_days'] = date_range['range_days'].replace(0, 1)
        
        user_features['transaction_frequency'] = (
            user_features['nb_transactions'] / date_range['range_days']
        ).replace([np.inf, -np.inf], 0).fillna(0).round(4)
        
        # Dernière transaction
        last_transaction = df.groupby('user_id')['transaction_date'].max()
        user_features['days_since_last_transaction'] = (
            pd.Timestamp.now() - last_transaction
        ).dt.days
        
        # Jour préféré pour les transactions
        def get_preferred_weekday(series):
            if len(series) > 0:
                return series.dt.dayofweek.mode()[0] if not series.dt.dayofweek.mode().empty else 0
            return 0
        
        preferred_weekday = df.groupby('user_id')['transaction_date'].apply(get_preferred_weekday)
        preferred_weekday.name = 'preferred_weekday'
        user_features = user_features.join(preferred_weekday)
        
        # Heure préférée
        df['hour'] = df['transaction_date'].dt.hour
        def get_preferred_hour(series):
            if len(series) > 0:
                return series.mode()[0] if not series.mode().empty else 12
            return 12
        
        preferred_hour = df.groupby('user_id')['hour'].apply(get_preferred_hour)
        preferred_hour.name = 'preferred_hour'
        user_features = user_features.join(preferred_hour)
    
    # Features monétaires supplémentaires
    if 'transaction_fee' in df.columns:
        fee_features = df.groupby('user_id')['transaction_fee'].agg([
            'sum', 'mean', 'max'
        ]).round(3)
        fee_features.columns = [
            'total_fees', 'avg_fee', 'max_fee'
        ]
        user_features = user_features.join(fee_features)
        
        # Ratio frais / montant
        user_features['fee_ratio'] = (
            user_features['total_fees'] / user_features['total_amount']
        ).replace([np.inf, -np.inf], 0).fillna(0).round(4)
    
    # Variabilité des montants
    user_features['amount_variability'] = (
        user_features['std_amount'] / user_features['avg_amount']
    ).replace([np.inf, -np.inf], 0).fillna(0).round(4)
    
    # APRÈS avoir créé toutes les features, gérer les colonnes catégorielles
    # Encodage one-hot pour top_category si présent
    if 'top_category' in user_features.columns:
        # Remplacer les valeurs manquantes par 'Unknown'
        user_features['top_category'] = user_features['top_category'].fillna('Unknown')
        
        # Limiter le nombre de catégories pour éviter trop de colonnes
        top_categories = user_features['top_category'].value_counts().head(10).index
        user_features['top_category_encoded'] = user_features['top_category'].apply(
            lambda x: x if x in top_categories else 'Other'
        )
        
        # One-hot encoding
        top_cat_dummies = pd.get_dummies(
            user_features['top_category_encoded'], 
            prefix='top_category', 
            drop_first=True
        )
        user_features = pd.concat([user_features, top_cat_dummies], axis=1)
        user_features = user_features.drop(['top_category', 'top_category_encoded'], axis=1)
    
    # Encodage one-hot pour top_payment_method si présent
    if 'top_payment_method' in user_features.columns:
        user_features['top_payment_method'] = user_features['top_payment_method'].fillna('Unknown')
        
        top_payments = user_features['top_payment_method'].value_counts().head(5).index
        user_features['top_payment_encoded'] = user_features['top_payment_method'].apply(
            lambda x: x if x in top_payments else 'Other'
        )
        
        payment_dummies = pd.get_dummies(
            user_features['top_payment_encoded'], 
            prefix='payment', 
            drop_first=True
        )
        user_features = pd.concat([user_features, payment_dummies], axis=1)
        user_features = user_features.drop(['top_payment_method', 'top_payment_encoded'], axis=1)
    
    # Convertir les autres colonnes catégorielles restantes en numériques
    categorical_cols = user_features.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        if col != 'top_category' and col != 'top_payment_method':  # Déjà traitées
            # Encodage ordinal simple pour les autres colonnes catégorielles
            user_features[col] = pd.factorize(user_features[col])[0]
    
    # Traitement des valeurs manquantes
    numeric_cols = user_features.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        # Remplacer inf/-inf par NaN d'abord
        user_features = user_features.replace([np.inf, -np.inf], np.nan)
        
        # Remplir les NaN
        for col in numeric_cols:
            if user_features[col].isna().any():
                # Utiliser la médiane pour les colonnes numériques
                user_features[col] = user_features[col].fillna(user_features[col].median())
    
    # Vérifier et supprimer les colonnes non-numériques restantes
    non_numeric_cols = user_features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"Suppression finale des colonnes non-numériques: {non_numeric_cols}")
        user_features = user_features.drop(columns=non_numeric_cols)
    
    # Vérifier qu'il reste des données
    if user_features.empty:
        raise ValueError("Aucune feature numérique créée!")
    
    # Vérifier les dimensions
    if user_features.shape[1] < 5:
        logger.warning(f"Peu de features créées: {user_features.shape[1]} colonnes")
    
    logger.info(f"Features utilisateur construites: {user_features.shape}")
    logger.info(f"Colonnes finales: {list(user_features.columns)}")
    
    return user_features


def get_user_feature_description() -> Dict[str, str]:
    """
    Retourne une description des features utilisateur.
    
    Returns:
        Dictionnaire {feature: description}
    """
    return {
        'nb_transactions': "Nombre total de transactions",
        'total_amount': "Montant total dépensé (€)",
        'avg_amount': "Montant moyen par transaction (€)",
        'std_amount': "Écart-type des montants (€)",
        'min_amount': "Montant minimum (€)",
        'max_amount': "Montant maximum (€)",
        'total_cashback': "Cashback total reçu (€)",
        'avg_cashback': "Cashback moyen par transaction (€)",
        'std_cashback': "Écart-type du cashback (€)",
        'max_cashback': "Cashback maximum reçu (€)",
        'cashback_ratio': "Ratio cashback/montant total (%)",
        'total_loyalty_points': "Points de fidélité totaux",
        'avg_loyalty_points': "Points de fidélité moyens par transaction",
        'max_loyalty_points': "Points de fidélité maximum par transaction",
        'nb_categories': "Nombre de catégories de produits différentes",
        'top_category_freq': "Fréquence de la catégorie préférée (%)",
        'nb_payment_methods': "Nombre de méthodes de paiement utilisées",
        'nb_device_types': "Nombre de types d'appareils utilisés",
        'nb_locations': "Nombre de localisations différentes",
        'transaction_frequency': "Fréquence moyenne des transactions (par jour)",
        'days_since_last_transaction': "Jours depuis la dernière transaction",
        'preferred_weekday': "Jour de la semaine préféré (0=lundi, 6=dimanche)",
        'preferred_hour': "Heure préférée pour les transactions (0-23)",
        'total_fees': "Total des frais de transaction (€)",
        'avg_fee': "Frais moyens par transaction (€)",
        'max_fee': "Frais maximum par transaction (€)",
        'fee_ratio': "Ratio frais/montant total (%)",
        'amount_variability': "Variabilité relative des montants (std/mean)",
    }


def get_user_features_stats(user_features: pd.DataFrame) -> pd.DataFrame:
    """
    Génère des statistiques descriptives pour les features utilisateur.
    
    Args:
        user_features: DataFrame des features utilisateur
        
    Returns:
        DataFrame avec statistiques
    """
    stats = user_features.describe().round(3).T
    stats['missing'] = user_features.isna().sum()
    stats['missing_pct'] = (stats['missing'] / len(user_features) * 100).round(2)
    stats['dtype'] = user_features.dtypes
    
    return stats


def filter_important_features(
    user_features: pd.DataFrame,
    correlation_threshold: float = 0.95,
    variance_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Filtre les features importantes en supprimant les colonnes:
    1. Trop corrélées entre elles
    2. Avec trop peu de variance
    
    Args:
        user_features: DataFrame des features
        correlation_threshold: Seuil de corrélation pour supprimer les colonnes
        variance_threshold: Seuil de variance minimum
        
    Returns:
        DataFrame filtré
    """
    df = user_features.copy()
    
    # 1. Supprimer les colonnes avec peu de variance
    variances = df.var()
    low_variance_cols = variances[variances < variance_threshold].index.tolist()
    
    if low_variance_cols:
        logger.info(f"Suppression des colonnes à faible variance: {low_variance_cols}")
        df = df.drop(columns=low_variance_cols)
    
    # 2. Supprimer les colonnes trop corrélées
    if len(df.columns) > 1:
        corr_matrix = df.corr().abs()
        
        # Matrice triangulaire supérieure
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Trouver les colonnes avec corrélation > threshold
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > correlation_threshold)]
        
        if to_drop:
            logger.info(f"Suppression des colonnes trop corrélées: {to_drop}")
            df = df.drop(columns=to_drop)
    
    logger.info(f"Features après filtrage: {df.shape}")
    return df