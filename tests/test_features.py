"""
Tests unitaires pour les fonctions de construction de features.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.user_features import build_user_features
from src.features.transaction_features import build_transaction_features


@pytest.fixture
def sample_transactions():
    """Crée un jeu de données de test."""
    np.random.seed(42)
    n_transactions = 100
    n_users = 20
    
    # Dates aléatoires sur les 30 derniers jours
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    dates = pd.date_range(start_date, end_date, periods=n_transactions)
    
    data = {
        'transaction_id': [f'tx_{i:03d}' for i in range(n_transactions)],
        'user_id': np.random.choice([f'user_{i:03d}' for i in range(n_users)], n_transactions),
        'transaction_date': np.random.choice(dates, n_transactions),
        'product_category': np.random.choice(['Electronics', 'Food', 'Clothing', 'Travel', 'Entertainment'], n_transactions),
        'product_amount': np.random.exponential(100, n_transactions).round(2),
        'transaction_fee': np.random.uniform(0, 5, n_transactions).round(2),
        'cashback': np.random.exponential(2, n_transactions).round(2),
        'loyalty_points': np.random.poisson(10, n_transactions),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'Wallet', 'Bank Transfer'], n_transactions),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_transactions),
        'location': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Lille', 'Bordeaux'], n_transactions)
    }
    
    # Ajouter quelques valeurs manquantes
    for col in ['cashback', 'loyalty_points']:
        mask = np.random.random(n_transactions) < 0.1
        data[col] = np.where(mask, np.nan, data[col])
    
    return pd.DataFrame(data)


def test_build_user_features(sample_transactions):
    """Test la construction des features utilisateur."""
    user_features = build_user_features(sample_transactions)
    
    # Vérifications de base
    assert isinstance(user_features, pd.DataFrame)
    assert len(user_features) > 0
    assert user_features.index.name == 'user_id'
    
    # Vérifier que certaines colonnes attendues sont présentes
    expected_cols = ['nb_transactions', 'total_amount', 'avg_amount']
    for col in expected_cols:
        assert col in user_features.columns
    
    # Vérifier qu'il n'y a pas de NaN dans les colonnes numériques
    numeric_cols = user_features.select_dtypes(include=[np.number]).columns
    assert not user_features[numeric_cols].isna().any().any()
    
    # Vérifier la cohérence des calculs
    user_id = user_features.index[0]
    user_transactions = sample_transactions[sample_transactions['user_id'] == user_id]
    
    assert user_features.loc[user_id, 'nb_transactions'] == len(user_transactions)
    assert abs(user_features.loc[user_id, 'total_amount'] - user_transactions['product_amount'].sum()) < 0.01


def test_build_transaction_features(sample_transactions):
    """Test la construction des features transaction."""
    tx_features = build_transaction_features(sample_transactions)
    
    # Vérifications de base
    assert isinstance(tx_features, pd.DataFrame)
    assert len(tx_features) == len(sample_transactions)
    assert len(tx_features.columns) > 0
    
    # Vérifier qu'il n'y a pas de NaN dans les colonnes numériques
    numeric_cols = tx_features.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        assert not tx_features[numeric_cols].isna().any().any()
    
    # Vérifier la présence de certaines colonnes attendues
    if 'product_amount' in sample_transactions.columns:
        assert 'product_amount' in tx_features.columns


def test_user_features_dimensions(sample_transactions):
    """Test les dimensions des features utilisateur."""
    user_features = build_user_features(sample_transactions)
    
    # Nombre d'utilisateurs unique
    n_unique_users = sample_transactions['user_id'].nunique()
    assert len(user_features) == n_unique_users
    
    # Vérifier que le nombre de features est raisonnable
    assert 5 <= len(user_features.columns) <= 30


def test_transaction_features_dimensions(sample_transactions):
    """Test les dimensions des features transaction."""
    tx_features = build_transaction_features(sample_transactions)
    
    # Même nombre de lignes que les données originales
    assert len(tx_features) == len(sample_transactions)
    
    # Vérifier que le nombre de features est raisonnable
    assert 5 <= len(tx_features.columns) <= 50


def test_missing_value_handling(sample_transactions):
    """Test la gestion des valeurs manquantes."""
    # Ajouter plus de valeurs manquantes
    df_with_nan = sample_transactions.copy()
    df_with_nan.loc[::10, 'cashback'] = np.nan
    df_with_nan.loc[::15, 'product_amount'] = np.nan
    
    user_features = build_user_features(df_with_nan)
    tx_features = build_transaction_features(df_with_nan)
    
    # Vérifier qu'il n'y a pas de NaN dans les résultats
    assert not user_features.select_dtypes(include=[np.number]).isna().any().any()
    assert not tx_features.select_dtypes(include=[np.number]).isna().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])