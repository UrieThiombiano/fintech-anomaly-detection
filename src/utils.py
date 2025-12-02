"""
Fonctions utilitaires génériques pour le projet.
"""
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


def validate_dataframe(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Valide qu'un DataFrame contient les colonnes attendues.
    
    Args:
        df: DataFrame à valider
        expected_columns: Liste des colonnes attendues
        
    Returns:
        True si toutes les colonnes attendues sont présentes
    """
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans les données: {missing_columns}"
        )
    return True


def handle_missing_values(df: pd.DataFrame, strategy: str = 'zero') -> pd.DataFrame:
    """
    Gère les valeurs manquantes selon une stratégie donnée.
    
    Args:
        df: DataFrame avec valeurs manquantes
        strategy: Stratégie de traitement ('zero', 'mean', 'median', 'drop')
        
    Returns:
        DataFrame traité
    """
    df_processed = df.copy()
    
    if strategy == 'zero':
        numeric_cols = df_processed.select_dtypes(include=np.number).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)
        
    elif strategy == 'mean':
        numeric_cols = df_processed.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            
    elif strategy == 'median':
        numeric_cols = df_processed.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
    elif strategy == 'drop':
        df_processed = df_processed.dropna()
        
    else:
        raise ValueError(f"Stratégie inconnue: {strategy}")
    
    return df_processed


def scale_features(
    X: Union[pd.DataFrame, np.ndarray],
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardise les features avec StandardScaler.
    
    Args:
        X: Données à standardiser
        scaler: Scaler existant (si None, en crée un nouveau)
        fit: Si True, fit le scaler sur les données
        
    Returns:
        Tuple (données standardisées, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
        
    return X_scaled, scaler


def save_model(model, path: str):
    """Sauvegarde un modèle avec joblib."""
    import joblib
    joblib.dump(model, path)


def load_model(path: str):
    """Charge un modèle sauvegardé avec joblib."""
    import joblib
    return joblib.load(path)