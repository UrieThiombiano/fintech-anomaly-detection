# 🔍 Fintech Anomaly Detection

Une application de data mining et machine learning non supervisé pour la détection d'anomalies dans les transactions fintech.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🎯 Objectifs du projet

Ce projet vise à analyser des transactions d'un portefeuille digital (wallet fintech) pour :

1. **Segmenter les utilisateurs** selon leurs comportements de dépenses
2. **Détecter des transactions anormales** (montants atypiques, abus de cashback, comportements suspects)
3. **Expliquer ces anomalies** avec des méthodes d'explicabilité (XAI)

## 🏗️ Architecture du projet
fintech-anomaly-detection/
├── app/ # Application Streamlit
├── src/ # Code source Python
├── data/ # Données (raw et processed)
├── models/ # Modèles sauvegardés
├── notebooks/ # Notebooks d'exploration
├── scripts/ # Scripts d'entraînement
├── tests/ # Tests unitaires
└── docs/ # Documentation


## 🚀 Installation rapide

### Option 1 : Installation locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/fintech-anomaly-detection.git
cd fintech-anomaly-detection

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Sur Windows :
venv\Scripts\activate
# Sur macOS/Linux :
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Lancer l'application
streamlit run app/streamlit_app.py


### Option 2 : Avec docker

# 1. Construire l'image Docker
docker build -t fintech-anomaly-detection .

# 2. Lancer le conteneur
docker run -p 8501:8501 fintech-anomaly-detection

# Ou avec docker-compose
docker-compose up -d

### Option 3 : Streamlit cloud

Forkez ce dépôt sur GitHub

Rendez-vous sur share.streamlit.io

Connectez votre compte GitHub

Sélectionnez le dépôt et le fichier app/streamlit_app.py

Cliquez sur "Deploy"

📊 Fonctionnalités de l'application
L'application Streamlit propose 6 pages principales :

1. 🎯 Objectifs du projet
Présentation du projet et de la méthodologie

Explication des méthodes utilisées

2. 🔍 Exploration des données
Aperçu des données brutes

Statistiques descriptives

Visualisation des distributions

Analyse des valeurs manquantes

3. 📊 ACP sur les utilisateurs
Analyse en Composantes Principales

Scree plot et variance expliquée

Représentation des individus

Cercle des corrélations

4. 👥 Segmentation KMeans
Méthode du coude pour choisir k

Scores de silhouette

Visualisation des clusters

Profils moyens par cluster

5. 🚨 Anomalies transactionnelles
Détection avec Isolation Forest

Distribution des scores d'anomalie

Filtrage par seuil

Liste des transactions suspectes

6. 🤖 Explications SHAP
Calcul des contributions par feature

Explication en français des anomalies

Diagrammes waterfall et bar plots

Importance globale des features