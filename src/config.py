"""
Configuration du projet - constantes et paramètres globaux.
"""
from pathlib import Path

# Chemins des répertoires
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# Création des répertoires si nécessaire
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Paramètres ML
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Paramètres par défaut des modèles
DEFAULT_N_CLUSTERS = 4
DEFAULT_CONTAMINATION = 0.02
DEFAULT_PCA_N_COMPONENTS = 3

# Colonnes attendues dans les données brutes - AJOUTEZ CES LIGNES
EXPECTED_COLUMNS = [
    'user_id', 'transaction_id', 'transaction_date',
    'product_category', 'product_amount', 'transaction_fee',
    'cashback', 'loyalty_points', 'payment_method',
    'device_type', 'location'
]

# Fichiers sauvegardés
USER_FEATURES_PATH = DATA_PROCESSED_DIR / "user_features.parquet"
TX_FEATURES_PATH = DATA_PROCESSED_DIR / "transaction_features.parquet"
KMEANS_MODEL_PATH = MODELS_ARTIFACTS_DIR / "kmeans_model.joblib"
ISOLATION_FOREST_PATH = MODELS_ARTIFACTS_DIR / "isolation_forest.joblib"