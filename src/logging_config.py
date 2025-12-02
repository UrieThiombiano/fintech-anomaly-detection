"""
Configuration du logging pour le projet.
"""
import logging
import sys
from pathlib import Path

# Création du logger principal
logger = logging.getLogger("fintech_anomaly")
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Handler console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Handler fichier (optionnel)
log_file = Path(__file__).resolve().parents[1] / "logs" / "app.log"
log_file.parent.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger enfant avec le nom spécifié."""
    return logger.getChild(name)