import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def compute_shap_for_isolation_forest(
    iforest: IsolationForest,
    scaler: StandardScaler,
    tx_features: pd.DataFrame,
    sample_size: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Calcule les valeurs SHAP pour un sous-échantillon de transactions
    pour expliquer les décisions de l'IsolationForest.

    Parameters
    ----------
    iforest : IsolationForest
        Modèle entraîné.
    scaler : StandardScaler
        Scaler associé au modèle.
    tx_features : pd.DataFrame
        Features transactionnelles (non standardisées).
    sample_size : int
        Taille de l'échantillon pour SHAP.
    random_state : int
        Graine de hasard.

    Returns
    -------
    dict
        Contient le sous-échantillon X_sample, shap_values et feature_names.
    """

    rng = np.random.RandomState(random_state)
    n = len(tx_features)
    sample_size = min(sample_size, n)

    sample_indices = rng.choice(n, size=sample_size, replace=False)
    X_sample = tx_features.iloc[sample_indices].copy()
    X_sample_scaled = scaler.transform(X_sample)

    explainer = shap.TreeExplainer(iforest)
    shap_values = explainer.shap_values(X_sample_scaled)

    result = {
        "X_sample": X_sample,
        "shap_values": shap_values,
        "feature_names": tx_features.columns.tolist(),
        "indices": sample_indices,
    }

    return result
