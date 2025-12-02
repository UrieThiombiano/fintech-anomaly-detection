from pathlib import Path
import pandas as pd
from .config import RAW_CSV_PATH


def load_raw_transactions(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Charge le fichier CSV de transactions brutes.

    Parameters
    ----------
    csv_path : Path | None
        Chemin vers le fichier CSV. Si None, utilise RAW_CSV_PATH.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les transactions.
    """
    if csv_path is None:
        csv_path = RAW_CSV_PATH

    df = pd.read_csv(csv_path)
    return df
