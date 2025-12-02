"""
Chargement et validation des données.
"""
import pandas as pd
import io
from typing import Optional, Union
from pathlib import Path

from src.config import EXPECTED_COLUMNS
from src.logging_config import get_logger
from src.utils import validate_dataframe

logger = get_logger(__name__)


def load_csv_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Charge un fichier CSV depuis le système de fichiers.
    
    Args:
        file_path: Chemin vers le fichier CSV
        
    Returns:
        DataFrame pandas
    """
    logger.info(f"Chargement du fichier CSV: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Validation des colonnes
    try:
        validate_dataframe(df, EXPECTED_COLUMNS)
        logger.info("Validation des colonnes réussie")
    except ValueError as e:
        logger.warning(f"Validation des colonnes: {e}")
        
    return df


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Charge un fichier uploadé via Streamlit.
    
    Args:
        uploaded_file: Fichier uploadé via st.file_uploader
        
    Returns:
        DataFrame pandas
    """
    logger.info("Chargement du fichier uploadé")
    file_bytes = uploaded_file.getvalue()
    df = pd.read_csv(io.BytesIO(file_bytes))
    logger.info(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Validation des colonnes
    try:
        validate_dataframe(df, EXPECTED_COLUMNS)
        logger.info("Validation des colonnes réussie")
    except ValueError as e:
        logger.warning(f"Validation des colonnes: {e}")
        
    return df


def load_raw_transactions(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Charge les données de transactions brutes.
    
    Args:
        file_path: Chemin optionnel vers le fichier
        
    Returns:
        DataFrame des transactions
    """
    from src.config import DATA_RAW_DIR
    
    if file_path is None:
        # Cherche le premier fichier CSV dans le répertoire raw
        csv_files = list(DATA_RAW_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {DATA_RAW_DIR}")
        file_path = csv_files[0]
    
    return load_csv_file(file_path)