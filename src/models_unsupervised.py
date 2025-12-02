from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from .config import MODELS_DIR, RANDOM_STATE


def fit_user_kmeans(
    user_features: pd.DataFrame,
    n_clusters: int = 4,
) -> dict:
    """
    Entraîne un pipeline StandardScaler + KMeans sur les features utilisateur.
    Sauvegarde le modèle dans MODELS_DIR.

    Returns
    -------
    dict
        Contient le scaler, le modèle KMeans et la PCA ajustée pour la visualisation.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_features)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init="auto",
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    model_bundle = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "cluster_labels": cluster_labels,
        "X_pca": X_pca,
    }

    joblib.dump(model_bundle, MODELS_DIR / "user_kmeans.joblib")

    return model_bundle


def fit_tx_isolation_forest(
    tx_features: pd.DataFrame,
    contamination: float = 0.02,
) -> dict:
    """
    Entraîne un pipeline StandardScaler + IsolationForest sur les transactions.
    Sauvegarde le modèle dans MODELS_DIR.

    Returns
    -------
    dict
        Contient le scaler, le modèle IsolationForest et les scores d'anomalie.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(tx_features)

    iforest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X_scaled)

    anomaly_score = -iforest.decision_function(X_scaled)
    is_anomaly = iforest.predict(X_scaled) == -1

    model_bundle = {
        "scaler": scaler,
        "iforest": iforest,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
    }

    joblib.dump(model_bundle, MODELS_DIR / "tx_iforest.joblib")

    return model_bundle
