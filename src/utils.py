"""
Fonctions utilitaires génériques pour le projet.
"""
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def validate_dataframe(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Valide qu'un DataFrame contient les colonnes attendues.
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
    """
    df_processed = df.copy()
    
    if strategy == 'zero':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)
        
    elif strategy == 'mean':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            
    elif strategy == 'median':
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
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
) -> Tuple[np.ndarray, StandardScaler, Optional[List[str]]]:
    """
    Standardise les features avec StandardScaler.
    
    Args:
        X: Données d'entrée
        scaler: Scaler existant (si None, en crée un nouveau)
        fit: Si True, fit le scaler sur les données
        
    Returns:
        Tuple (données standardisées, scaler, noms des colonnes)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    # Si X est un DataFrame, extraire uniquement les colonnes numériques
    if isinstance(X, pd.DataFrame):
        # Garder les noms des colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            raise ValueError("Aucune colonne numérique trouvée pour la standardisation")
        
        X_numeric = X[numeric_cols].values
        
        if fit:
            X_scaled = scaler.fit_transform(X_numeric)
        else:
            X_scaled = scaler.transform(X_numeric)
            
        return X_scaled, scaler, numeric_cols
    else:
        # Si déjà un array numpy
        if fit:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
            
        return X_scaled, scaler, None


def prepare_numeric_data(
    df: pd.DataFrame,
    drop_non_numeric: bool = True,
    fill_na: bool = True
) -> pd.DataFrame:
    """
    Prépare les données pour les algorithmes ML qui nécessitent des données numériques.
    
    Args:
        df: DataFrame d'entrée
        drop_non_numeric: Si True, supprime les colonnes non-numériques
        fill_na: Si True, remplit les valeurs manquantes
        
    Returns:
        DataFrame préparé
    """
    df_clean = df.copy()
    
    # Supprimer les colonnes non-numériques si demandé
    if drop_non_numeric:
        non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"Suppression des colonnes non-numériques: {list(non_numeric_cols)}")
            df_clean = df_clean.drop(columns=non_numeric_cols)
    
    # Remplir les valeurs manquantes
    if fill_na and not df_clean.empty:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    return df_clean


def encode_categorical_columns(
    df: pd.DataFrame,
    max_categories: int = 10
) -> pd.DataFrame:
    """
    Encode les colonnes catégorielles en numériques.
    
    Args:
        df: DataFrame avec colonnes catégorielles
        max_categories: Nombre maximum de catégories pour one-hot encoding
        
    Returns:
        DataFrame avec colonnes encodées
    """
    df_encoded = df.copy()
    
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        unique_count = df_encoded[col].nunique()
        
        if unique_count <= max_categories:
            # One-hot encoding pour peu de catégories
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
        else:
            # Encodage ordinal pour beaucoup de catégories
            df_encoded[col] = pd.factorize(df_encoded[col])[0]
    
    return df_encoded


def save_model(model, path: str):
    """Sauvegarde un modèle avec joblib."""
    import joblib
    joblib.dump(model, path)


def load_model(path: str):
    """Charge un modèle sauvegardé avec joblib."""
    import joblib
    return joblib.load(path)


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Vérifie la qualité des données.
    
    Returns:
        Dictionnaire avec métriques de qualité
    """
    quality = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_values': df.isna().sum().sum(),
        'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum()
    }
    
    return quality