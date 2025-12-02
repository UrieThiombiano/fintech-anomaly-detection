"""
Script d'entraînement principal pour l'application hors ligne.
"""
import sys
from pathlib import Path

# Ajouter le répertoire src au path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np
from datetime import datetime

from src.data.loader import load_raw_transactions
from src.features.user_features import build_user_features
from src.features.transaction_features import build_transaction_features
from src.models.clustering import train_kmeans
from src.models.anomaly_detection import train_isolation_forest
from src.xai.shap_explainer import compute_shap_for_isolation_forest
from src.config import DATA_PROCESSED_DIR, MODELS_ARTIFACTS_DIR
from src.utils import save_model


def main():
    """Pipeline complet d'entraînement."""
    print("=" * 60)
    print("🚀 Pipeline d'entraînement - Détection d'anomalies Fintech")
    print("=" * 60)
    
    # 1. Chargement des données
    print("\n1. 📥 Chargement des données...")
    df_raw = load_raw_transactions()
    print(f"   ✓ {df_raw.shape[0]} transactions chargées")
    print(f"   ✓ {df_raw.shape[1]} colonnes")
    
    # 2. Construction des features
    print("\n2. 🛠️ Construction des features...")
    
    print("   Features utilisateur...")
    user_features = build_user_features(df_raw)
    print(f"   ✓ {user_features.shape[0]} utilisateurs")
    print(f"   ✓ {user_features.shape[1]} features utilisateur")
    
    print("   Features transaction...")
    tx_features = build_transaction_features(df_raw)
    print(f"   ✓ {tx_features.shape[0]} transactions")
    print(f"   ✓ {tx_features.shape[1]} features transaction")
    
    # Sauvegarde des features
    user_features.to_parquet(DATA_PROCESSED_DIR / "user_features.parquet")
    tx_features.to_parquet(DATA_PROCESSED_DIR / "transaction_features.parquet")
    print(f"   ✓ Features sauvegardées dans {DATA_PROCESSED_DIR}")
    
    # 3. Segmentation des utilisateurs
    print("\n3. 👥 Segmentation des utilisateurs (KMeans)...")
    kmeans_result = train_kmeans(user_features, n_clusters=4)
    
    # Sauvegarde du modèle KMeans
    save_model(kmeans_result, MODELS_ARTIFACTS_DIR / "kmeans_model.joblib")
    print(f"   ✓ KMeans entraîné avec {kmeans_result['n_clusters']} clusters")
    print(f"   ✓ Score silhouette: {kmeans_result['silhouette_score']:.3f}")
    
    # 4. Détection d'anomalies
    print("\n4. 🚨 Détection d'anomalies (Isolation Forest)...")
    iforest_result = train_isolation_forest(tx_features, contamination=0.02)
    
    # Sauvegarde du modèle Isolation Forest
    save_model(iforest_result, MODELS_ARTIFACTS_DIR / "isolation_forest.joblib")
    print(f"   ✓ Isolation Forest entraîné")
    print(f"   ✓ {iforest_result['is_anomaly'].sum()} anomalies détectées")
    
    # 5. Calcul SHAP
    print("\n5. 🤖 Calcul des explications SHAP...")
    shap_result = compute_shap_for_isolation_forest(
        iforest_result['iforest'],
        iforest_result['scaler'],
        tx_features,
        sample_size=200
    )
    
    # Sauvegarde SHAP
    np.savez_compressed(
        MODELS_ARTIFACTS_DIR / "shap_results.npz",
        shap_values=shap_result['shap_values'],
        indices=shap_result['indices'],
        feature_names=shap_result['feature_names']
    )
    print(f"   ✓ SHAP calculé sur {shap_result['sample_size']} transactions")
    
    # 6. Résumé
    print("\n" + "=" * 60)
    print("✅ Pipeline terminé avec succès!")
    print("=" * 60)
    
    print(f"\n📊 Résumé des résultats:")
    print(f"   • Utilisateurs: {user_features.shape[0]}")
    print(f"   • Transactions: {tx_features.shape[0]}")
    print(f"   • Clusters: {kmeans_result['n_clusters']}")
    print(f"   • Anomalies détectées: {iforest_result['is_anomaly'].sum()}")
    print(f"   • Taux d'anomalies: {iforest_result['is_anomaly'].sum()/len(tx_features)*100:.1f}%")
    print(f"\n💾 Modèles sauvegardés dans: {MODELS_ARTIFACTS_DIR}")
    
    # Génération d'un rapport simple
    generate_report(df_raw, user_features, tx_features, kmeans_result, iforest_result)


def generate_report(df_raw, user_features, tx_features, kmeans_result, iforest_result):
    """Génère un rapport d'analyse simple."""
    report_path = MODELS_ARTIFACTS_DIR / f"rapport_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("📊 RAPPORT D'ANALYSE - Détection d'anomalies Fintech\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 📥 DONNÉES\n")
        f.write(f"   • Transactions brutes: {df_raw.shape[0]}\n")
        f.write(f"   • Colonnes brutes: {df_raw.shape[1]}\n")
        f.write(f"   • Utilisateurs uniques: {user_features.shape[0]}\n")
        f.write(f"   • Période couverte: {df_raw['transaction_date'].min()} au {df_raw['transaction_date'].max()}\n\n")
        
        f.write("2. 🛠️ FEATURES\n")
        f.write(f"   • Features utilisateur: {user_features.shape[1]}\n")
        f.write(f"   • Features transaction: {tx_features.shape[1]}\n\n")
        
        f.write("3. 👥 SEGMENTATION UTILISATEURS\n")
        f.write(f"   • Nombre de clusters: {kmeans_result['n_clusters']}\n")
        f.write(f"   • Score silhouette: {kmeans_result['silhouette_score']:.3f}\n")
        
        # Taille des clusters
        unique, counts = np.unique(kmeans_result['cluster_labels'], return_counts=True)
        for cluster, count in zip(unique, counts):
            f.write(f"   • Cluster {cluster}: {count} utilisateurs ({count/len(user_features)*100:.1f}%)\n")
        
        f.write("\n4. 🚨 DÉTECTION D'ANOMALIES\n")
        f.write(f"   • Contamination paramétrée: {iforest_result['contamination']}\n")
        f.write(f"   • Anomalies détectées: {iforest_result['is_anomaly'].sum()}\n")
        f.write(f"   • Taux d'anomalies: {iforest_result['is_anomaly'].sum()/len(tx_features)*100:.1f}%\n")
        f.write(f"   • Score moyen d'anomalie: {iforest_result['anomaly_scores'].mean():.3f}\n\n")
        
        f.write("5. 📍 FICHIERS GÉNÉRÉS\n")
        f.write(f"   • Features utilisateur: {DATA_PROCESSED_DIR}/user_features.parquet\n")
        f.write(f"   • Features transaction: {DATA_PROCESSED_DIR}/transaction_features.parquet\n")
        f.write(f"   • Modèle KMeans: {MODELS_ARTIFACTS_DIR}/kmeans_model.joblib\n")
        f.write(f"   • Modèle Isolation Forest: {MODELS_ARTIFACTS_DIR}/isolation_forest.joblib\n")
        f.write(f"   • Résultats SHAP: {MODELS_ARTIFACTS_DIR}/shap_results.npz\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"📅 Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60)
    
    print(f"\n📄 Rapport généré: {report_path}")


if __name__ == "__main__":
    main()