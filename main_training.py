from pathlib import Path
import pandas as pd

from src.config import DATA_PROCESSED_DIR, USER_FEATURES_PATH, TX_FEATURES_PATH
from src.data_loading import load_raw_transactions
from src.features_user import build_user_features
from src.features_tx import build_transaction_features
from src.models_unsupervised import fit_user_kmeans, fit_tx_isolation_forest
from src.shap_utils import compute_shap_for_isolation_forest


def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_transactions()
    print(f"Transactions brutes : {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes")

    user_features = build_user_features(df_raw)
    print(f"Features utilisateur : {user_features.shape}")

    tx_features = build_transaction_features(df_raw)
    print(f"Features transaction : {tx_features.shape}")

    user_features.to_parquet(USER_FEATURES_PATH, index=True)
    tx_features.to_parquet(TX_FEATURES_PATH, index=False)
    print(f"Features sauvegardées dans {DATA_PROCESSED_DIR}")

    user_model_bundle = fit_user_kmeans(user_features, n_clusters=4)
    print("Modèle KMeans utilisateur entraîné et sauvegardé.")

    tx_model_bundle = fit_tx_isolation_forest(tx_features, contamination=0.02)
    print("Modèle IsolationForest transaction entraîné et sauvegardé.")

    shap_result = compute_shap_for_isolation_forest(
        iforest=tx_model_bundle["iforest"],
        scaler=tx_model_bundle["scaler"],
        tx_features=tx_features,
        sample_size=200,
    )
    print("SHAP calculé sur un sous-échantillon de transactions.")

    shap_output_path = DATA_PROCESSED_DIR / "shap_tx_sample.npz"
    import numpy as np

    np.savez_compressed(
        shap_output_path,
        shap_values=shap_result["shap_values"],
        indices=shap_result["indices"],
        feature_names=shap_result["feature_names"],
    )
    print(f"Résultats SHAP sauvegardés dans {shap_output_path}")


if __name__ == "__main__":
    main()
