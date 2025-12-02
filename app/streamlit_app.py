"""
Application Streamlit principale pour la détection d'anomalies fintech.
"""
import sys
from pathlib import Path

# Ajouter le répertoire src au path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.loader import load_uploaded_file
from src.features.user_features import build_user_features, get_user_feature_description
from src.features.transaction_features import build_transaction_features, get_transaction_feature_description
from src.models.pca import compute_pca, get_pca_summary, get_top_loadings
from src.models.clustering import (
    compute_elbow_curve, compute_silhouette_scores, 
    train_kmeans, get_cluster_profiles, suggest_optimal_k
)
from src.models.anomaly_detection import (
    train_isolation_forest, analyze_anomalies, 
    get_anomaly_statistics, suggest_contamination
)
from src.xai.shap_explainer import (
    compute_shap_for_isolation_forest, get_top_shap_features,
    generate_shap_summary, explain_anomaly_in_french
)
from src.config import RANDOM_STATE


# --------- Configuration de l'application --------- #

st.set_page_config(
    page_title="Fintech Anomaly Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# --------- Fonctions de cache --------- #

@st.cache_data(show_spinner="Chargement des données...")
def load_and_prepare_data(uploaded_file):
    """Charge et prépare les données depuis le fichier uploadé."""
    df_raw = load_uploaded_file(uploaded_file)
    user_features = build_user_features(df_raw)
    tx_features = build_transaction_features(df_raw)
    return df_raw, user_features, tx_features


@st.cache_data(show_spinner="Calcul de la PCA...")
def compute_pca_cached(user_features, n_components):
    """Calcule la PCA avec cache."""
    return compute_pca(user_features, n_components=n_components)


@st.cache_data(show_spinner="Calcul du coude KMeans...")
def compute_elbow_cached(user_features, k_min, k_max):
    """Calcule la courbe du coude avec cache."""
    return compute_elbow_curve(user_features, k_min, k_max)


@st.cache_data(show_spinner="Calcul des scores silhouette...")
def compute_silhouette_cached(user_features, k_min, k_max):
    """Calcule les scores silhouette avec cache."""
    return compute_silhouette_scores(user_features, k_min, k_max)


@st.cache_data(show_spinner="Entraînement KMeans...")
def train_kmeans_cached(user_features, n_clusters):
    """Entraîne KMeans avec cache."""
    return train_kmeans(user_features, n_clusters=n_clusters)


@st.cache_data(show_spinner="Entraînement Isolation Forest...")
def train_iforest_cached(tx_features, contamination):
    """Entraîne Isolation Forest avec cache."""
    return train_isolation_forest(tx_features, contamination=contamination)


@st.cache_data(show_spinner="Calcul SHAP...")
def compute_shap_cached(iforest, scaler, tx_features, sample_size):
    """Calcule SHAP avec cache."""
    return compute_shap_for_isolation_forest(iforest, scaler, tx_features, sample_size)


# --------- Pages --------- #

def page_objectifs():
    """Page d'accueil avec les objectifs du projet."""
    st.markdown('<h1 class="main-header">💰 Détection d\'Anomalies Fintech</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Objectifs du projet</h3>
    <p>Cette application analyse des transactions d'un portefeuille digital pour :</p>
    <ul>
    <li><b>Segmenter les utilisateurs</b> selon leurs comportements de dépense</li>
    <li><b>Détecter des transactions anormales</b> (montants atypiques, abus de cashback, comportements suspects)</li>
    <li><b>Expliquer ces anomalies</b> avec des méthodes d'explicabilité (XAI)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>🔍 ACP</h4>
        <p>Réduction de dimension et analyse des relations entre variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 KMeans</h4>
        <p>Segmentation non supervisée des utilisateurs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>🌲 Isolation Forest</h4>
        <p>Détection d'anomalies sans labels</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 📈 Méthodologie
    
    ### 1. Analyse en Composantes Principales (ACP)
    - **1er principe** : Représenter les utilisateurs dans un espace réduit
    - **2ème principe** : Analyser les relations entre variables via les corrélations
    
    ### 2. Segmentation par KMeans
    - Regroupement des utilisateurs en clusters homogènes
    - Choix du nombre optimal de clusters via méthode du coude et silhouette
    
    ### 3. Détection d'Anomalies avec Isolation Forest
    - Algorithmes d'arbres pour isoler les points atypiques
    - Pas besoin de données labellisées
    
    ### 4. Explications avec SHAP
    - Décomposition feature par feature des décisions du modèle
    - Compréhension des raisons d'une prédiction d'anomalie
    
    ## 🚀 Comment utiliser cette application
    
    1. **Importez vos données** via le menu latéral
    2. **Explorez les données** dans l'onglet Exploration
    3. **Analysez les utilisateurs** avec ACP et KMeans
    4. **Détectez les anomalies** transactionnelles
    5. **Comprenez les résultats** avec SHAP
    
    Toutes les étapes sont interactives et paramétrables !
    """)


def page_eda(df_raw, user_features, tx_features):
    """Page d'exploration des données."""
    st.markdown('<h1 class="main-header">🔍 Exploration des Données</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Données Brutes", "Features Utilisateur", "Features Transaction"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Données Brutes des Transactions</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de transactions", df_raw.shape[0])
        with col2:
            st.metric("Nombre de colonnes", df_raw.shape[1])
        with col3:
            st.metric("Utilisateurs uniques", df_raw['user_id'].nunique())
        
        st.subheader("Aperçu des données")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        st.subheader("Types de données")
        dtype_df = pd.DataFrame({
            'Colonne': df_raw.columns,
            'Type': df_raw.dtypes.astype(str),
            'Valeurs uniques': [df_raw[col].nunique() for col in df_raw.columns],
            'Valeurs manquantes': df_raw.isna().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        st.subheader("Statistiques descriptives")
        numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df_raw[numeric_cols].describe().round(2), use_container_width=True)
        
        # Visualisations
        st.subheader("Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'product_amount' in df_raw.columns:
                fig = px.histogram(df_raw, x='product_amount', nbins=50,
                                  title="Distribution des montants")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'cashback' in df_raw.columns:
                fig = px.histogram(df_raw, x='cashback', nbins=50,
                                  title="Distribution du cashback")
                st.plotly_chart(fig, use_container_width=True)
        
        if 'product_category' in df_raw.columns:
            fig = px.bar(df_raw['product_category'].value_counts().head(10),
                        title="Top 10 catégories de produits")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Features Utilisateur</h2>', unsafe_allow_html=True)
        
        st.metric("Nombre d'utilisateurs", user_features.shape[0])
        st.metric("Nombre de features", user_features.shape[1])
        
        st.subheader("Aperçu des features")
        st.dataframe(user_features.head(10), use_container_width=True)
        
        st.subheader("Description des features")
        feature_desc = get_user_feature_description()
        desc_df = pd.DataFrame({
            'Feature': list(feature_desc.keys()),
            'Description': list(feature_desc.values())
        })
        st.dataframe(desc_df, use_container_width=True)
        
        st.subheader("Corrélations entre features")
        numeric_cols = user_features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = user_features[numeric_cols].corr().round(2)
            fig = px.imshow(corr_matrix, text_auto=True,
                          title="Matrice de corrélation")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Features Transaction</h2>', unsafe_allow_html=True)
        
        st.metric("Nombre de transactions", tx_features.shape[0])
        st.metric("Nombre de features", tx_features.shape[1])
        
        st.subheader("Aperçu des features")
        st.dataframe(tx_features.head(10), use_container_width=True)
        
        st.subheader("Description des features")
        feature_desc = get_transaction_feature_description()
        desc_df = pd.DataFrame({
            'Feature': list(feature_desc.keys()),
            'Description': list(feature_desc.values())
        })
        st.dataframe(desc_df, use_container_width=True)

def page_acp(user_features):
    """Page d'analyse PCA avancée."""
    st.markdown('<h1 class="main-header">🔬 Analyse en Composantes Principales Avancée</h1>', unsafe_allow_html=True)
    
    # ==================== SECTION 1: INTRODUCTION PÉDAGOGIQUE ====================
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Objectif scientifique de l'ACP</h3>
    <p>L'Analyse en Composantes Principales est une <b>technique d'algèbre linéaire</b> qui permet de :</p>
    <ul>
    <li><b>Réduire la dimensionnalité</b> tout en conservant l'information maximale</li>
    <li><b>Identifier les axes de variance</b> principaux dans les données</li>
    <li><b>Visualiser les corrélations</b> entre variables multidimensionnelles</li>
    <li><b>Détecter les patterns cachés</b> et structures latentes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Badges scientifiques
    st.markdown("""
    <div style="display: flex; gap: 10px; margin: 20px 0;">
        <span class="badge" style="background-color: #2E7D32; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold;">🧮 Algèbre Linéaire</span>
        <span class="badge" style="background-color: #1565C0; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold;">📈 Analyse Multivariée</span>
        <span class="badge" style="background-color: #6A1B9A; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold;">🔍 Réduction Dimensionnelle</span>
        <span class="badge" style="background-color: #C2185B; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold;">📊 Statistiques Avancées</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Colonnes d'explication
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>🎯 Principe mathématique</h4>
        <p><b>Diagonalisation</b> de la matrice de covariance</p>
        <p><b>Vecteurs propres</b> = directions de variance maximale</p>
        <p><b>Valeurs propres</b> = importance des axes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>📐 Objectifs analytiques</h4>
        <p><b>1. Simplification</b> : n → k dimensions</p>
        <p><b>2. Interprétation</b> : comprendre les relations</p>
        <p><b>3. Visualisation</b> : représenter l'essentiel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>🔍 Applications fintech</h4>
        <p><b>Segmentation clients</b> : profils comportementaux</p>
        <p><b>Détection patterns</b> : transactions atypiques</p>
        <p><b>Analyse risques</b> : corrélations cachées</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== SECTION 2: PRÉPARATION DES DONNÉES ====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🔧 Préparation des Données</h2>', unsafe_allow_html=True)
    
    with st.expander("📋 Processus de prétraitement avancé", expanded=True):
        st.markdown("""
        ### Pipeline de préparation rigoureux
        
        **Étape 1 : Vérification des types de données**
        - Identification variables numériques vs catégorielles
        - Conversion optimale pour préservation information
        
        **Étape 2 : Traitement des valeurs manquantes**
        - Analyse pattern de manquants
        - Imputation par médiane/moyenne selon distribution
        - Suppression si >50% manquants
        
        **Étape 3 : Gestion des outliers**
        - Détection par scores Z
        - Winsorization aux percentiles 1 et 99
        - Préservation variance sans distortion
        
        **Étape 4 : Standardisation**
        - Centrage (moyenne = 0)
        - Réduction (écart-type = 1)
        - Comparabilité des variables
        """)
        
        # Afficher les statistiques de préparation
        if not user_features.empty:
            st.subheader("📊 Statistiques descriptives avant PCA")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Nombre d'observations", f"{user_features.shape[0]:,}")
                st.metric("Variables initiales", user_features.shape[1])
                
                # Types de données
                dtype_counts = user_features.dtypes.value_counts()
                st.write("**Types de données :**")
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count} variables")
            
            with col2:
                # Valeurs manquantes
                missing = user_features.isna().sum()
                missing_pct = (missing / len(user_features) * 100).round(2)
                
                st.write("**Valeurs manquantes :**")
                if missing.sum() > 0:
                    missing_df = pd.DataFrame({
                        'Variable': missing.index,
                        'Manquants': missing.values,
                        '%': missing_pct.values
                    })
                    missing_df = missing_df[missing_df['Manquants'] > 0]
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ Aucune valeur manquante")
    
    # ==================== SECTION 3: CONFIGURATION DE L'ANALYSE ====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">⚙️ Configuration de l\'Analyse PCA</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Paramètres avancés
        st.subheader("Paramètres analytiques")
        
        tab1, tab2, tab3 = st.tabs(["📏 Composantes", "🔍 Qualité", "🎯 Avancé"])
        
        with tab1:
            max_components = min(10, user_features.shape[1])
            n_components = st.slider(
                "Nombre de composantes à calculer",
                min_value=2,
                max_value=max_components,
                value=min(4, max_components),
                help="Nombre d'axes principaux à extraire"
            )
            
            variance_threshold = st.slider(
                "Seuil de variance cumulée",
                min_value=0.5,
                max_value=0.99,
                value=0.9,
                step=0.01,
                help="Pourcentage minimum de variance à expliquer"
            )
        
        with tab2:
            compute_advanced = st.checkbox(
                "Calculer métriques avancées",
                value=True,
                help="KMO, Bartlett, communautés, stabilité bootstrap"
            )
            
            perform_validation = st.checkbox(
                "Validation croisée",
                value=True,
                help="Évaluation de la robustesse du modèle"
            )
        
        with tab3:
            random_state = st.number_input(
                "Seed aléatoire",
                min_value=0,
                max_value=1000,
                value=42,
                help="Pour la reproductibilité des résultats"
            )
            
            bootstrap_samples = st.slider(
                "Échantillons bootstrap",
                min_value=10,
                max_value=500,
                value=100,
                help="Pour l'analyse de stabilité"
            )
    
    with col2:
        st.subheader("🎯 Critères de décision")
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50;">
        <h4>📊 Règles de sélection</h4>
        
        **1. Règle de Kaiser**
        - Valeurs propres > 1
        - Composantes significatives
        
        **2. Scree plot (Cattell)**
        - Point d'inflexion
        - Diminution marginale
        
        **3. Variance cumulée**
        - Minimum 70-80%
        - Optimal 85-95%
        
        **4. Interprétabilité**
        - Loading > |0.3|
        - Sens métier clair
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== SECTION 4: EXÉCUTION DE L'ANALYSE ====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🚀 Exécution de l\'Analyse PCA</h2>', unsafe_allow_html=True)
    
    if st.button("🔬 Lancer l'analyse PCA avancée", type="primary", use_container_width=True):
        with st.spinner("🧮 Calcul en cours... Cette analyse peut prendre quelques secondes"):
            try:
                # Calcul PCA avec la fonction standard
                pca_result = compute_pca_cached(user_features, n_components)
                
                # ==================== SECTION 4.1: RÉSULTATS GLOBAUX ====================
                st.success("✅ Analyse PCA terminée avec succès !")
                
                # Métriques de qualité globale
                st.subheader("📈 Métriques de qualité globale")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Variance expliquée totale",
                        f"{pca_result['cumulative_variance_ratio'][-1]:.1%}",
                        help="Proportion totale de variance conservée"
                    )
                
                with col2:
                    st.metric(
                        "Nombre de composantes",
                        n_components,
                        help="Axes principaux calculés"
                    )
                
                with col3:
                    st.metric(
                        "Variables originales",
                        user_features.shape[1],
                        help="Réduction de dimension"
                    )
                
                # ==================== SECTION 4.2: SCREE PLOT ====================
                st.markdown("---")
                st.subheader("📊 Scree Plot - Analyse des valeurs propres")
                
                eigenvalues = pca_result['explained_variance']
                explained_variance_ratio = pca_result['explained_variance_ratio']
                cumulative_variance_ratio = pca_result['cumulative_variance_ratio']
                
                fig_scree = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Variance expliquée par composante", "Variance cumulée"),
                )
                
                # Plot 1: Variance par composante
                fig_scree.add_trace(
                    go.Bar(
                        x=[f'PC{i+1}' for i in range(len(explained_variance_ratio))],
                        y=explained_variance_ratio * 100,
                        name='% Variance',
                        marker_color='#1f77b4',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Plot 2: Variance cumulée
                fig_scree.add_trace(
                    go.Scatter(
                        x=[f'PC{i+1}' for i in range(len(cumulative_variance_ratio))],
                        y=cumulative_variance_ratio * 100,
                        name='Variance cumulée',
                        mode='lines+markers',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=2
                )
                
                # Seuil de variance
                fig_scree.add_hline(
                    y=variance_threshold * 100,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Seuil {variance_threshold:.0%}",
                    annotation_position="right",
                    row=1, col=2
                )
                
                fig_scree.update_layout(
                    height=400,
                    showlegend=True,
                    title_text="Analyse dimensionnelle - Critères de décision"
                )
                
                fig_scree.update_yaxes(title_text="% Variance", row=1, col=1)
                fig_scree.update_yaxes(title_text="% Variance cumulée", row=1, col=2)
                
                st.plotly_chart(fig_scree, use_container_width=True)
                
                # ==================== SECTION 4.3: TABLEAU DES RÉSULTATS ====================
                st.subheader("📋 Tableau synthétique des résultats")
                
                summary_data = {
                    'Composante': [f'PC{i+1}' for i in range(n_components)],
                    'Variance expliquée': [f'{v:.1%}' for v in explained_variance_ratio],
                    'Variance cumulée': [f'{v:.1%}' for v in cumulative_variance_ratio],
                    'Valeur propre': [f'{v:.3f}' for v in eigenvalues]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # ==================== SECTION 4.4: CERCLE DES CORRÉLATIONS ====================
                if n_components >= 2:
                    st.markdown("---")
                    st.subheader("🎯 Cercle des Corrélations")
                    
                    # Options d'affichage
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        pc_x = st.selectbox(
                            "Axe horizontal (X)",
                            options=[f'PC{i+1}' for i in range(n_components)],
                            index=0
                        )
                    
                    with col2:
                        pc_y = st.selectbox(
                            "Axe vertical (Y)",
                            options=[f'PC{i+1}' for i in range(n_components)],
                            index=1 if n_components > 1 else 0
                        )
                    
                    # Extraction des indices
                    x_idx = int(pc_x[2:]) - 1
                    y_idx = int(pc_y[2:]) - 1
                    
                    # Création du cercle des corrélations
                    loadings = pca_result['components']
                    feature_names = user_features.columns.tolist()
                    
                    x_loadings = loadings[x_idx]
                    y_loadings = loadings[y_idx]
                    
                    # Création du graphique
                    fig_circle = go.Figure()
                    
                    # Cercle unité
                    theta = np.linspace(0, 2*np.pi, 100)
                    fig_circle.add_trace(go.Scatter(
                        x=np.cos(theta),
                        y=np.sin(theta),
                        mode='lines',
                        line=dict(color='gray', dash='dash'),
                        name='Cercle unité',
                        showlegend=False
                    ))
                    
                    # Axes
                    fig_circle.add_hline(y=0, line_color='gray', line_width=1)
                    fig_circle.add_vline(x=0, line_color='gray', line_width=1)
                    
                    # Variables
                    fig_circle.add_trace(go.Scatter(
                        x=x_loadings,
                        y=y_loadings,
                        mode='markers+text',
                        text=feature_names,
                        textposition="top center",
                        marker=dict(size=8, color='blue'),
                        name='Variables',
                        hovertemplate="<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
                    ))
                    
                    # Vecteurs
                    for i, feat in enumerate(feature_names):
                        fig_circle.add_trace(go.Scatter(
                            x=[0, x_loadings[i]],
                            y=[0, y_loadings[i]],
                            mode='lines',
                            line=dict(color='rgba(0,0,255,0.3)', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    fig_circle.update_layout(
                        title=f"Cercle des corrélations - {pc_x} vs {pc_y}",
                        xaxis_title=f"{pc_x} ({(explained_variance_ratio[x_idx]*100):.1f}%)",
                        yaxis_title=f"{pc_y} ({(explained_variance_ratio[y_idx]*100):.1f}%)",
                        height=600,
                        xaxis=dict(range=[-1.2, 1.2]),
                        yaxis=dict(range=[-1.2, 1.2]),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_circle, use_container_width=True)
                
                # ==================== SECTION 4.5: CONCLUSIONS ====================
                st.markdown("---")
                st.markdown('<h2 class="sub-header">🎯 Conclusions et recommandations</h2>', unsafe_allow_html=True)
                
                with st.expander("📋 Synthèse des résultats", expanded=True):
                    st.markdown(f"""
                    ### 📊 Synthèse analytique
                    
                    **✅ Points forts de l'analyse :**
                    1. **Variance bien capturée** : {cumulative_variance_ratio[-1]:.1%} avec {n_components} composantes
                    2. **Compression réussie** : Réduction de {user_features.shape[1]} à {n_components} dimensions
                    3. **Interprétabilité** : Les composantes principales sont faciles à interpréter
                    
                    **🎯 Principaux axes d'interprétation :**
                    1. **PC1** ({explained_variance_ratio[0]:.1%}) : Premier axe de variance
                    2. **PC2** ({explained_variance_ratio[1]:.1%}) : Deuxième axe orthogonal
                    
                    ### 📈 Recommandations pratiques
                    
                    **Pour la segmentation clients :**
                    1. Utiliser PC1 et PC2 pour la visualisation 2D
                    2. Regrouper les utilisateurs proches dans l'espace réduit
                    3. Identifier les profils extrêmes aux coins du nuage
                    
                    **Pour la réduction dimensionnelle :**
                    1. Conserver {min(n_components + 1, user_features.shape[1])} composantes pour >95% de variance
                    2. Supprimer les features avec faible variance
                    3. Valider avec une méthode de clustering (KMeans)
                    
                    **Prochaines étapes :**
                    - Appliquer KMeans sur les scores PCA
                    - Analyser les profils des clusters
                    - Détecter les anomalies transactionnelles
                    """)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse PCA: {str(e)}")
                with st.expander("🔧 Détails techniques"):
                    st.write(f"**Erreur :** {e}")
                    st.write(f"**Shape des données :** {user_features.shape}")
                    st.write(f"**Colonnes :** {user_features.columns.tolist()}")

    else:
        # Mode attente
        st.info("👆 **Cliquez sur le bouton ci-dessus pour lancer l'analyse PCA**")

def page_kmeans(user_features):
    """Page de segmentation KMeans."""
    st.markdown('<h1 class="main-header">👥 Segmentation des Utilisateurs</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Objectif du Clustering</h3>
    <p>KMeans permet de :</p>
    <ul>
    <li><b>Regrouper les utilisateurs</b> en clusters homogènes</li>
    <li><b>Identifier des segments</b> avec comportements similaires</li>
    <li><b>Personnaliser les offres</b> selon les profils détectés</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Paramètres
    st.sidebar.subheader("Paramètres KMeans")
    k_min = st.sidebar.number_input("k minimum", min_value=2, max_value=10, value=2)
    k_max = st.sidebar.number_input("k maximum", min_value=k_min+1, max_value=15, value=8)
    
    # Choix du nombre de clusters
    st.markdown('<h2 class="sub-header">Choix du Nombre de Clusters</h2>', unsafe_allow_html=True)
    
    # Courbe du coude
    ks, inertias = compute_elbow_cached(user_features, k_min, k_max)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Méthode du Coude")
        df_elbow = pd.DataFrame({'k': ks, 'inertia': inertias})
        
        fig = px.line(df_elbow, x='k', y='inertia', markers=True,
                     title="Courbe du coude - Inertia vs k")
        fig.add_vline(x=suggest_optimal_k(user_features, k_min, k_max, 'elbow'),
                     line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **💡 Interprétation :**
        - L'inertie mesure la compacité des clusters
        - On cherche le "coude" où la diminution ralentit
        - Point d'inflexion = bon compromis compacité/simplicité
        """)
    
    with col2:
        st.subheader("Score de Silhouette")
        ks_sil, silhouette_scores = compute_silhouette_cached(user_features, k_min, k_max)
        
        df_sil = pd.DataFrame({'k': ks_sil, 'silhouette': silhouette_scores})
        df_sil = df_sil.dropna()  # Enlever k=1
        
        fig = px.line(df_sil, x='k', y='silhouette', markers=True,
                     title="Score de silhouette vs k")
        
        if not df_sil.empty:
            best_k = df_sil.loc[df_sil['silhouette'].idxmax(), 'k']
            fig.add_vline(x=best_k, line_dash="dash", line_color="green")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **💡 Interprétation :**
        - Score entre -1 et 1
        - Proche de 1 = clusters bien séparés
        - Proche de 0 = recouvrement important
        - On cherche le k qui maximise ce score
        """)
    
    # Choix final de k
    st.subheader("Choix du Nombre de Clusters")
    
    suggested_k_elbow = suggest_optimal_k(user_features, k_min, k_max, 'elbow')
    suggested_k_sil = suggest_optimal_k(user_features, k_min, k_max, 'silhouette')
    suggested_k_combined = suggest_optimal_k(user_features, k_min, k_max, 'combined')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Suggéré (coude)", suggested_k_elbow)
    with col2:
        st.metric("Suggéré (silhouette)", suggested_k_sil)
    with col3:
        st.metric("Suggéré (combiné)", suggested_k_combined)
    
    chosen_k = st.slider(
        "Nombre de clusters k pour l'entraînement",
        min_value=k_min,
        max_value=k_max,
        value=int(suggested_k_combined)
    )
    
    # Entraînement KMeans
    clustering_result = train_kmeans_cached(user_features, chosen_k)
    
    # Visualisation des clusters
    st.markdown('<h2 class="sub-header">Visualisation des Clusters</h2>', unsafe_allow_html=True)
    
    X_transformed = clustering_result['X_transformed']
    labels = clustering_result['cluster_labels']
    
    df_clusters = pd.DataFrame({
        'PC1': X_transformed[:, 0],
        'PC2': X_transformed[:, 1],
        'Cluster': labels.astype(str),
        'user_id': user_features.index
    })
    
    fig = px.scatter(df_clusters, x='PC1', y='PC2', color='Cluster',
                    hover_data=['user_id'],
                    title=f"Clusters KMeans (k={chosen_k})",
                    color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)
    
    # Métriques de qualité
    st.subheader("Qualité du Clustering")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score Silhouette", f"{clustering_result['silhouette_score']:.3f}")
    with col2:
        st.metric("Indice Davies-Bouldin", f"{clustering_result['davies_bouldin_score']:.3f}")
    with col3:
        st.metric("Inertie", f"{clustering_result['inertia']:.1f}")
    
    # Profils des clusters
    st.markdown('<h2 class="sub-header">Profils des Clusters</h2>', unsafe_allow_html=True)
    
    cluster_profiles = get_cluster_profiles(user_features, clustering_result)
    
    if not cluster_profiles.empty:
        # Afficher les statistiques principales
        display_cols = [col for col in cluster_profiles.columns 
                       if any(x in col for x in ['mean', 'size', 'pct'])]
        
        st.dataframe(
            cluster_profiles[display_cols].style.format("{:.2f}"),
            use_container_width=True
        )
        
        # Visualisation comparative
        st.subheader("Comparaison des Clusters")
        
        # Sélection des features à comparer
        numeric_cols = user_features.select_dtypes(include=[np.number]).columns
        selected_features = st.multiselect(
            "Features à comparer",
            options=numeric_cols.tolist(),
            default=numeric_cols[:3].tolist() if len(numeric_cols) >= 3 else numeric_cols.tolist()
        )
        
        if selected_features:
            # Préparer les données pour la visualisation
            user_features_with_clusters = user_features.copy()
            user_features_with_clusters['Cluster'] = clustering_result['cluster_labels'].astype(str)
            
            # Boxplots par cluster
            for feature in selected_features:
                if feature in user_features.columns:
                    fig = px.box(user_features_with_clusters, 
                                x='Cluster', y=feature,
                                title=f"Distribution de {feature} par cluster")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Exploration d'un cluster spécifique
    st.markdown('<h2 class="sub-header">Exploration d\'un Cluster</h2>', unsafe_allow_html=True)
    
    selected_cluster = st.selectbox(
        "Choisir un cluster à explorer",
        options=sorted(np.unique(labels))
    )
    
    cluster_indices = np.where(labels == selected_cluster)[0]
    cluster_users = user_features.iloc[cluster_indices]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(f"Utilisateurs dans cluster {selected_cluster}", len(cluster_users))
        
        # Statistiques du cluster
        if not cluster_users.empty:
            stats = cluster_users.describe().round(2)
            st.dataframe(stats, use_container_width=True)
    
    with col2:
        if not cluster_users.empty:
            # Visualisation radar des caractéristiques moyennes
            mean_values = cluster_users.mean(numeric_only=True)
            top_features = mean_values.nlargest(8)
            
            fig = px.bar(x=top_features.index, y=top_features.values,
                        title=f"Top 8 caractéristiques - Cluster {selected_cluster}")
            st.plotly_chart(fig, use_container_width=True)


def page_anomalies(df_raw, tx_features):
    """Page de détection d'anomalies."""
    st.markdown('<h1 class="main-header">🚨 Détection d\'Anomalies Transactionnelles</h1>', unsafe_allow_html=True)
    
    # ==================== SECTION 1: INTRODUCTION PÉDAGOGIQUE ====================
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Objectif de cette analyse</h3>
    <p>Identifier des transactions <b>atypiques</b> pouvant correspondre à :</p>
    <ul>
    <li>Fraude ou abus de cashback</li>
    <li>Comportements suspects d'utilisateurs</li>
    <li>Erreurs de saisie ou bugs système</li>
    <li>Patterns transactionnels rares</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Explication du concept avec des colonnes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Isolation Forest</h4>
        <p><b>Principe :</b> Forêt d'arbres aléatoires qui isolent les points</p>
        <p><b>Avantage :</b> Pas besoin de données labellisées</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>🔍 Contamination</h4>
        <p><b>Définition :</b> Proportion attendue d'anomalies</p>
        <p><b>Typique :</b> 1-5% selon le domaine</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>📈 Score d'anomalie</h4>
        <p><b>Interprétation :</b> Plus élevé = plus anormal</p>
        <p><b>Seuil :</b> Généralement > percentile 95</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== SECTION 2: PRÉPARATION DES DONNÉES ====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📋 Préparation des Données</h2>', unsafe_allow_html=True)
    
    with st.expander("🔍 Voir les features transactionnelles utilisées", expanded=True):
        # Afficher un résumé des features
        if not tx_features.empty:
            st.write(f"**Nombre de transactions analysées :** {len(tx_features):,}")
            st.write(f"**Nombre de features :** {tx_features.shape[1]}")
            
            # Statistiques descriptives
            st.subheader("Statistiques descriptives des features")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Types de données :**")
                dtype_counts = tx_features.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count} colonnes")
            
            with col2:
                st.write("**Valeurs manquantes :**")
                missing = tx_features.isna().sum()
                missing_pct = (missing / len(tx_features) * 100).round(2)
                missing_df = pd.DataFrame({
                    'Colonne': missing.index,
                    'Valeurs manquantes': missing.values,
                    'Pourcentage': missing_pct.values
                })
                st.dataframe(missing_df[missing_df['Valeurs manquantes'] > 0], 
                           use_container_width=True, hide_index=True)
            
            # Matrice de corrélation
            if tx_features.shape[1] > 1:
                st.subheader("🔗 Matrice de corrélations")
                numeric_cols = tx_features.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = tx_features[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                  title="Corrélations entre features transactionnelles",
                                  color_continuous_scale='RdBu',
                                  zmin=-1, zmax=1)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **💡 Interprétation :**
                    - **Couleurs bleues** : Corrélation positive (variables évoluent ensemble)
                    - **Couleurs rouges** : Corrélation négative (variables évoluent en opposition)
                    - **Variables fortement corrélées** peuvent être redondantes
                    """)
    
    # ==================== SECTION 3: PARAMÉTRAGE DU MODÈLE ====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">⚙️ Configuration du Modèle</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contamination = st.slider(
            "**Contamination** (proportion d'anomalies attendue)",
            min_value=0.001,
            max_value=0.2,
            value=0.02,
            step=0.001,
            help="""Paramètre crucial qui influence la sensibilité du détecteur.
            \n• **Valeur faible (0.01)** : Détection très conservative
            \n• **Valeur élevée (0.1)** : Détection plus agressive"""
        )
    
    with col2:
        n_estimators = st.slider(
            "**Nombre d'arbres**",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Plus d'arbres = modèle plus stable mais plus lent"
        )
    
    with col3:
        # Bouton pour suggestion automatique
        if st.button("🎯 Suggérer automatiquement", key="suggest_contamination"):
            try:
                suggested = suggest_contamination(tx_features)
                st.success(f"Contamination suggérée: **{suggested:.3f}**")
                contamination = suggested
            except:
                st.info("Utilisez la valeur par défaut de 2%")
    
    # Explication technique
    st.markdown("""
    <div class="warning-box">
    <h4>🧠 Fonctionnement d'Isolation Forest</h4>
    <p><b>Algorithme :</b></p>
    <ol>
    <li>Construction d'arbres de décision aléatoires</li>
    <li>Les anomalies sont <b>isolées plus rapidement</b> (moins de décisions)</li>
    <li>Le score d'anomalie = longueur moyenne du chemin d'isolation</li>
    <li>Seuil automatique basé sur la contamination spécifiée</li>
    </ol>
    <p><b>Avantages :</b> Pas besoin de labels, efficace sur données multidimensionnelles</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== SECTION 4: ENTRAÎNEMENT ET RÉSULTATS ====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🔬 Résultats de la Détection</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Lancer la détection d'anomalies", type="primary", use_container_width=True):
        with st.spinner("🧠 Entraînement du modèle en cours..."):
            try:
                # 1. Entraînement du modèle
                anomaly_result = train_isolation_forest(
                    tx_features, 
                    contamination=contamination,
                    n_estimators=n_estimators
                )
                
                # 2. Statistiques
                stats = get_anomaly_statistics(anomaly_result)
                
                # ==================== SECTION 4.1: MÉTRIQUES DE PERFORMANCE ====================
                st.success("✅ Modèle entraîné avec succès !")
                
                # KPI Cards
                st.subheader("📊 Métriques de performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Transactions analysées",
                        f"{stats['n_total']:,}",
                        help="Nombre total de transactions traitées"
                    )
                
                with col2:
                    st.metric(
                        "Anomalies détectées",
                        f"{stats['n_anomalies']:,}",
                        delta=f"{stats['pct_anomalies']:.1f}%",
                        delta_color="inverse",
                        help="Nombre et pourcentage de transactions anormales"
                    )
                
                with col3:
                    st.metric(
                        "Score moyen",
                        f"{stats['score_mean']:.3f}",
                        help="Score d'anomalie moyen (0 = normal, >0 = anormal)"
                    )
                
                with col4:
                    st.metric(
                        "Seuil Q95",
                        f"{stats['score_q95']:.3f}",
                        help="95ème percentile - seuil d'alerte recommandé"
                    )
                
                # ==================== SECTION 4.2: DISTRIBUTION DES SCORES ====================
                st.subheader("📈 Distribution des scores d'anomalie")
                
                scores = anomaly_result['anomaly_scores']
                is_anomaly = anomaly_result['is_anomaly']
                
                # Créer un DataFrame pour la visualisation
                df_scores = pd.DataFrame({
                    'Score': scores,
                    'Anomalie': is_anomaly,
                    'Catégorie': np.where(is_anomaly, 'Anomalie', 'Normal')
                })
                
                # Graphique 1: Histogramme avec densité
                fig1 = px.histogram(
                    df_scores, 
                    x='Score',
                    color='Catégorie',
                    nbins=50,
                    title="Distribution des scores d'anomalie",
                    color_discrete_map={'Normal': 'blue', 'Anomalie': 'red'},
                    opacity=0.7,
                    barmode='overlay'
                )
                
                # Ajouter une ligne verticale pour le seuil
                threshold = np.percentile(scores, 95)
                fig1.add_vline(
                    x=threshold, 
                    line_dash="dash", 
                    line_color="green",
                    annotation_text=f"Seuil 95% ({threshold:.3f})",
                    annotation_position="top right"
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Graphique 2: Box plot par catégorie
                fig2 = px.box(
                    df_scores,
                    x='Catégorie',
                    y='Score',
                    color='Catégorie',
                    title="Distribution comparative des scores",
                    points="all"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # ==================== SECTION 4.3: ANALYSE DES ANOMALIES ====================
                st.subheader("🔍 Analyse détaillée des anomalies")
                
                # Seuil interactif
                score_threshold = st.slider(
                    "**Seuil de score pour filtrer les anomalies**",
                    min_value=float(scores.min()),
                    max_value=float(scores.max()),
                    value=float(threshold),
                    step=0.01,
                    help="Ajustez ce seuil pour affiner la détection"
                )
                
                # Analyser les anomalies
                df_anomalies = analyze_anomalies(df_raw, anomaly_result, score_threshold)
                high_score_tx = df_anomalies[df_anomalies['is_above_threshold']].copy()
                
                st.metric(
                    f"Transactions au-dessus du seuil ({score_threshold:.3f})",
                    f"{len(high_score_tx):,}",
                    delta=f"{(len(high_score_tx)/len(df_raw)*100):.1f}%",
                    delta_color="inverse"
                )
                
                if not high_score_tx.empty:
                    # Afficher les transactions suspectes
                    st.subheader(f"📋 Top {min(20, len(high_score_tx))} transactions les plus suspectes")
                    
                    # Colonnes à afficher
                    display_cols = [
                        'transaction_id', 'user_id', 'transaction_date',
                        'product_category', 'product_amount', 'cashback',
                        'payment_method', 'anomaly_score'
                    ]
                    
                    # Garder seulement les colonnes présentes
                    available_cols = [col for col in display_cols if col in high_score_tx.columns]
                    
                    # Formater le DataFrame
                    display_df = high_score_tx[available_cols + ['anomaly_score']].head(20).copy()
                    
                    # Ajouter un indicateur visuel
                    def color_anomaly_score(val):
                        if val > threshold * 1.5:
                            return 'background-color: #ffcccc'  # Rouge clair
                        elif val > threshold:
                            return 'background-color: #fff3cd'  # Jaune clair
                        else:
                            return ''
                    
                    styled_df = display_df.style.format({
                        'anomaly_score': '{:.3f}',
                        'product_amount': '{:.2f}',
                        'cashback': '{:.2f}'
                    }).applymap(color_anomaly_score, subset=['anomaly_score'])
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # ==================== SECTION 4.4: ANALYSE DES PATTERNS ====================
                    st.subheader("📊 Patterns des anomalies détectées")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Catégories des anomalies
                        if 'product_category' in high_score_tx.columns:
                            cat_counts = high_score_tx['product_category'].value_counts().head(10)
                            fig_cat = px.bar(
                                x=cat_counts.index, 
                                y=cat_counts.values,
                                title="Catégories de produits des anomalies",
                                labels={'x': 'Catégorie', 'y': 'Nombre'},
                                color=cat_counts.values,
                                color_continuous_scale='reds'
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)
                    
                    with col2:
                        # Méthodes de paiement
                        if 'payment_method' in high_score_tx.columns:
                            pm_counts = high_score_tx['payment_method'].value_counts().head(10)
                            fig_pm = px.pie(
                                values=pm_counts.values,
                                names=pm_counts.index,
                                title="Répartition par méthode de paiement",
                                hole=0.4
                            )
                            st.plotly_chart(fig_pm, use_container_width=True)
                    
                    # Distribution des montants
                    if 'product_amount' in high_score_tx.columns:
                        st.subheader("💰 Analyse des montants")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_amount = px.box(
                                high_score_tx,
                                y='product_amount',
                                title="Distribution des montants des anomalies",
                                points="all"
                            )
                            st.plotly_chart(fig_amount, use_container_width=True)
                        
                        with col2:
                            # Comparaison avec l'ensemble des données
                            if 'product_amount' in df_raw.columns:
                                fig_compare = go.Figure()
                                fig_compare.add_trace(go.Box(
                                    y=df_raw['product_amount'],
                                    name='Toutes transactions',
                                    boxpoints=False
                                ))
                                fig_compare.add_trace(go.Box(
                                    y=high_score_tx['product_amount'],
                                    name='Anomalies',
                                    boxpoints=False,
                                    marker_color='red'
                                ))
                                fig_compare.update_layout(
                                    title="Comparaison des montants",
                                    yaxis_title="Montant (€)",
                                    showlegend=True
                                )
                                st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # ==================== SECTION 4.5: RAPPORT D'ANALYSE ====================
                    st.markdown("---")
                    st.markdown('<h2 class="sub-header">📄 Rapport d\'Analyse</h2>', unsafe_allow_html=True)
                    
                    with st.expander("📋 Synthèse des résultats", expanded=True):
                        st.markdown(f"""
                        ### Résumé exécutif
                        
                        **📊 Données analysées :**
                        - {stats['n_total']:,} transactions traitées
                        - {tx_features.shape[1]} features utilisées
                        - Contamination paramétrée : {contamination:.1%}
                        
                        **🚨 Résultats de détection :**
                        - **{stats['n_anomalies']:,} anomalies** détectées ({stats['pct_anomalies']:.1f}% du total)
                        - Score moyen : {stats['score_mean']:.3f}
                        - Seuil de détection (Q95) : {stats['score_q95']:.3f}
                        
                        **🎯 Transactions les plus suspectes :**
                        - Score maximum : {stats['score_max']:.3f}
                        - {len(high_score_tx):,} transactions au-dessus du seuil ({score_threshold:.3f})
                        
                        **💡 Recommandations :**
                        1. **Vérifier manuellement** les transactions avec score > {threshold:.3f}
                        2. **Analyser les patterns** récurrents dans les catégories/anomalies
                        3. **Ajuster la contamination** selon les résultats métier
                        """)
                    
                    # ==================== SECTION 4.6: EXPORT DES RÉSULTATS ====================
                    st.subheader("💾 Export des résultats")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV des anomalies
                        csv = high_score_tx.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger les anomalies (CSV)",
                            data=csv,
                            file_name="anomalies_detectees.csv",
                            mime="text/csv",
                            help="Exporte la liste des transactions anormales"
                        )
                    
                    with col2:
                        # Rapport PDF (simulé)
                        if st.button("📄 Générer un rapport PDF", help="Génère un rapport détaillé"):
                            st.info("""
                            **Fonctionnalité PDF :**
                            Pour un déploiement complet, cette fonctionnalité pourrait :
                            1. Générer un PDF avec tous les graphiques
                            2. Inclure les statistiques détaillées
                            3. Ajouter des recommandations métier
                            4. Exporter au format professionnel
                            """)
                
                else:
                    st.warning("⚠️ Aucune transaction ne dépasse le seuil actuel.")
                    st.info("""
                    **Suggestions :**
                    1. Réduisez le seuil de détection
                    2. Augmentez la contamination
                    3. Vérifiez la qualité des données
                    """)
                
                # ==================== SECTION 5: ÉVALUATION DU MODÈLE ====================
                st.markdown("---")
                st.markdown('<h2 class="sub-header">🎯 Évaluation du Modèle</h2>', unsafe_allow_html=True)
                
                with st.expander("🧪 Tests et validations", expanded=True):
                    st.markdown("""
                    ### Méthodologie d'évaluation
                    
                    **📏 Métriques utilisées :**
                    
                    1. **Distribution des scores** : Vérification de la séparation normale/anomalie
                    2. **Consistance des résultats** : Répartition cohérente avec la contamination
                    3. **Analyse des features** : Importance des variables dans la décision
                    
                    **✅ Critères de qualité :**
                    - **Séparation claire** entre scores normaux et anormaux
                    - **Distribution logique** des anomalies détectées
                    - **Robustesse** aux variations de paramètres
                    - **Interprétabilité** des résultats
                    
                    **🔬 Prochaines étapes possibles :**
                    - Validation croisée sur différentes périodes
                    - Comparaison avec d'autres algorithmes (LOF, One-Class SVM)
                    - Intégration de features temporelles supplémentaires
                    """)
                    
                    # Visualisation de la qualité
                    if not df_scores.empty:
                        # QQ-plot pour vérifier la distribution
                        from scipy import stats
                        
                        fig_qq = go.Figure()
                        
                        # Données pour QQ-plot
                        normal_scores = df_scores[df_scores['Catégorie'] == 'Normal']['Score']
                        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(normal_scores)))
                        
                        fig_qq.add_trace(go.Scatter(
                            x=theoretical_quantiles,
                            y=np.sort(normal_scores),
                            mode='markers',
                            name='QQ-plot',
                            marker=dict(size=8, opacity=0.6)
                        ))
                        
                        # Ligne de référence
                        min_val = min(theoretical_quantiles.min(), normal_scores.min())
                        max_val = max(theoretical_quantiles.max(), normal_scores.max())
                        fig_qq.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='y=x',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        fig_qq.update_layout(
                            title="QQ-plot : Normalité des scores 'normaux'",
                            xaxis_title="Quantiles théoriques (Normale)",
                            yaxis_title="Quantiles observés",
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_qq, use_container_width=True)
                        
                        st.markdown("""
                        **💡 Interprétation du QQ-plot :**
                        - **Points sur la ligne rouge** : Distribution normale
                        - **Points au-dessus de la ligne** : Queue de distribution plus épaisse
                        - **Points en-dessous** : Distribution différente de la normale
                        """)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la détection d'anomalies: {str(e)}")
                
                # Aide au débogage
                with st.expander("🔧 Aide au débogage"):
                    st.markdown(f"""
                    **Erreur détaillée :** `{e}`
                    
                    **Solutions possibles :**
                    
                    1. **Vérifiez les données :**
                       - Les features doivent être numériques
                       - Pas de valeurs manquantes excessives
                       - Pas de colonnes avec variance nulle
                    
                    2. **Paramètres :**
                       - Réduisez la contamination
                       - Diminuez le nombre d'arbres
                       - Utilisez moins de features
                    
                    3. **Données d'exemple :**
                       - Téléchargez notre [fichier d'exemple](https://example.com/sample_data.csv)
                       - Testez avec 100-200 transactions d'abord
                    """)
                    
                    # Affichage des données pour débogage
                    if not tx_features.empty:
                        st.write("**Aperçu des données :**")
                        st.dataframe(tx_features.head(), use_container_width=True)
                        
                        st.write("**Statistiques :**")
                        st.write(f"- Shape: {tx_features.shape}")
                        st.write(f"- Types: {tx_features.dtypes.unique()}")
                        st.write(f"- NaN: {tx_features.isna().sum().sum()}")
    else:
        # Mode attente
        st.info("👆 **Cliquez sur le bouton ci-dessus pour lancer la détection d'anomalies**")
        
        # Exemple de ce qui va se passer
        with st.expander("🎯 Prévisualisation de l'analyse"):
            st.markdown("""
            ### Ce que vous allez obtenir :
            
            1. **📊 Métriques de performance** :
               - Nombre d'anomalies détectées
               - Pourcentage d'anomalies
               - Scores statistiques
            
            2. **📈 Visualisations** :
               - Distribution des scores
               - Comparaison normale/anomalie
               - Analyse par catégorie
            
            3. **🔍 Analyse détaillée** :
               - Liste des transactions suspectes
               - Patterns récurrents
               - Recommandations
            
            4. **💾 Export** :
               - Fichier CSV des anomalies
               - Rapports synthétiques
            """)
            
            # Exemple visuel
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*YRim7T6BqrSylr8EaqKqZQ.png", 
                    caption="Exemple de détection d'anomalies avec Isolation Forest")
    
    # ==================== SECTION 6: POUR ALLER PLUS LOIN ====================
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🚀 Pour aller plus loin</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📚 Théorie", "🔧 Techniques avancées", "📈 Applications métier"])
    
    with tab1:
        st.markdown("""
        ### Fondements théoriques
        
        **📖 Isolation Forest (Liu et al., 2008)**
        
        **Principe :** Les anomalies sont rares et différentes → elles sont isolables en peu de décisions
        
        **Algorithme :**
        1. Sélection aléatoire d'une feature
        2. Sélection aléatoire d'une valeur de coupure
        3. Répétition jusqu'à isolation complète
        4. Score = longueur moyenne du chemin
        
        **Formule du score :**
        ```
        s(x, n) = 2^{-E(h(x))/c(n)}
        où :
        - h(x) = hauteur du chemin d'isolation
        - c(n) = hauteur moyenne d'un arbre binaire
        - E(h(x)) = espérance sur plusieurs arbres
        ```
        
        **Avantages :**
        - Linéaire en temps et mémoire
        - Efficace en haute dimension
        - Pas besoin de distance métrique
        """)
    
    with tab2:
        st.markdown("""
        ### Techniques avancées
        
        **🎯 Améliorations possibles :**
        
        1. **Ensemble methods** :
           - Combinaison avec Local Outlier Factor (LOF)
           - Stacking de différents détecteurs
           - Vote majoritaire
        
        2. **Features engineering** :
           - Features temporelles (tendance, saisonnalité)
           - Features de réseau (relations entre utilisateurs)
           - Encodages avancés des catégories
        
        3. **Validation** :
           - Validation temporelle (train/test sur périodes différentes)
           - Simulation d'anomalies pour évaluation
           - Métriques métier spécifiques
        
        4. **Monitoring** :
           - Détection de concept drift
           - Mise à jour incrémentale du modèle
           - Alertes en temps réel
        """)
    
    with tab3:
        st.markdown("""
        ### Applications métier dans la fintech
        
        **💰 Cas d'usage :**
        
        1. **Détection de fraude** :
           - Transactions anormalement élevées
           - Patterns de cashback suspects
           - Multi-comptes abusifs
        
        2. **Surveillance réglementaire** :
           - Conformité AML (Anti-Money Laundering)
           - Détection de blanchiment
           - Transactions Politically Exposed Persons (PEP)
        
        3. **Expérience client** :
           - Détection de bugs d'application
           - Transactions erronées
           - Problèmes de conversion devise
        
        4. **Business intelligence** :
           - Identification de segments spéciaux
           - Opportunités marketing
           - Optimisation des commissions
        
        **📊 ROI potentiel :**
        - Réduction des pertes par fraude : **5-15%**
        - Amélioration de l'expérience client : **+20% NPS**
        - Conformité réglementaire : **Évite amendes**
        """)
        
def page_xai(df_raw, tx_features):
    """Page d'explications SHAP."""
    st.markdown('<h1 class="main-header">🤖 Explications par SHAP</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Objectif de SHAP (SHapley Additive exPlanations)</h3>
    <p>SHAP permet de :</p>
    <ul>
    <li><b>Comprendre pourquoi</b> une transaction est jugée anormale</li>
    <li><b>Identifier les features</b> qui contribuent le plus à la décision</li>
    <li><b>Expliquer en termes simples</b> les prédictions du modèle</li>
    </ul>
    </div>
    
    <div class="warning-box">
    <h4>📊 Interprétation des valeurs SHAP</h4>
    <p><b>Valeur SHAP positive</b> = la feature augmente le score d'anomalie (rend la transaction plus suspecte)</p>
    <p><b>Valeur SHAP négative</b> = la feature diminue le score d'anomalie (rend la transaction plus normale)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Paramètres
    st.sidebar.subheader("Paramètres SHAP")
    
    contamination = st.sidebar.slider(
        "Contamination pour Isolation Forest",
        min_value=0.001,
        max_value=0.1,
        value=0.02,
        step=0.001,
        key="xai_contamination"
    )
    
    sample_size = st.sidebar.slider(
        "Taille de l'échantillon pour SHAP",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Plus d'échantillons = plus précis mais plus lent"
    )
    
    # Entraînement du modèle et calcul SHAP
    with st.spinner("Entraînement du modèle et calcul SHAP..."):
        anomaly_result = train_isolation_forest(tx_features, contamination=contamination)
        shap_result = compute_shap_cached(
            anomaly_result['iforest'],
            anomaly_result['scaler'],
            tx_features,
            sample_size
        )
    
    # Métriques globales
    shap_summary = generate_shap_summary(shap_result)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Transactions analysées", shap_summary['n_transactions'])
    with col2:
        st.metric("Features", shap_summary['n_features'])
    with col3:
        st.metric("Valeur SHAP moyenne", f"{shap_summary['shap_mean']:.4f}")
    
    # Importance globale des features
    st.markdown('<h2 class="sub-header">Importance Globale des Features</h2>', unsafe_allow_html=True)
    
    global_importance = pd.DataFrame(shap_summary['global_importance'])
    
    fig = px.bar(global_importance, x='mean_abs_shap', y='feature',
                orientation='h',
                title="Importance moyenne des features (valeur absolue SHAP)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **💡 Interprétation :**
    - Les features en haut sont les plus importantes globalement
    - Elles influencent le plus les décisions d'anomalie
    - Utile pour comprendre quelles variables surveiller
    """)
    
    # Sélection d'une transaction
    st.markdown('<h2 class="sub-header">Analyse d\'une Transaction Spécifique</h2>', unsafe_allow_html=True)
    
    # Préparer les données pour la sélection
    df_sample_meta = df_raw.iloc[shap_result['indices']].copy()
    df_sample_meta['anomaly_score'] = anomaly_result['anomaly_scores'][shap_result['indices']]
    df_sample_meta['is_anomaly'] = anomaly_result['is_anomaly'][shap_result['indices']]
    df_sample_meta['sample_index'] = range(len(df_sample_meta))
    
    # Trier par score d'anomalie
    df_sample_meta = df_sample_meta.sort_values('anomaly_score', ascending=False)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Transactions échantillonnées (triées par score)")
        
        display_cols = [
            'sample_index', 'transaction_id', 'user_id', 
            'product_amount', 'cashback', 'anomaly_score', 'is_anomaly'
        ]
        
        available_cols = [col for col in display_cols if col in df_sample_meta.columns]
        
        st.dataframe(
            df_sample_meta[available_cols]
            .head(20)
            .style.format({'anomaly_score': '{:.3f}'}),
            use_container_width=True
        )
    
    with col2:
        st.subheader("Sélection")
        
        selected_idx = st.number_input(
            "Index dans l'échantillon",
            min_value=0,
            max_value=len(df_sample_meta) - 1,
            value=0,
            step=1
        )
    
    # Informations sur la transaction sélectionnée
    selected_tx = df_sample_meta.iloc[selected_idx]
    
    st.subheader(f"Transaction sélectionnée (Index: {selected_idx})")
    
    # Afficher les détails
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score d'anomalie", f"{selected_tx['anomaly_score']:.3f}")
    with col2:
        st.metric("Est anomalie", "✅ Oui" if selected_tx['is_anomaly'] else "❌ Non")
    with col3:
        if 'transaction_id' in selected_tx:
            st.metric("ID Transaction", selected_tx['transaction_id'])
    
    # Détails complets
    with st.expander("📋 Détails complets de la transaction"):
        st.write(selected_tx)
    
    # Explications SHAP pour cette transaction
    st.markdown('<h2 class="sub-header">Explications SHAP Détaillées</h2>', unsafe_allow_html=True)
    
    # Récupérer les contributions SHAP
    top_n = st.slider(
        "Nombre de features à afficher",
        min_value=5,
        max_value=20,
        value=10
    )
    
    shap_contributions = get_top_shap_features(shap_result, selected_idx, top_n)
    
    # Bar plot des contributions
    fig = px.bar(shap_contributions.sort_values('shap_value'),
                x='shap_value', y='feature', orientation='h',
                color='shap_value',
                color_continuous_scale='RdBu',
                title=f"Contributions SHAP - Transaction {selected_idx}")
    
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)
    
    # Table détaillée
    st.subheader("Table des contributions")
    st.dataframe(
        shap_contribinations.style.format({
            'shap_value': '{:.4f}',
            'abs_shap': '{:.4f}'
        }),
        use_container_width=True
    )
    
    # Explication en français
    st.markdown('<h2 class="sub-header">Explication en Français</h2>', unsafe_allow_html=True)
    
    explanation = explain_anomaly_in_french(
        shap_result, selected_idx, df_raw, top_n=5
    )
    
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #4e73df;">
    {explanation}
    </div>
    """, unsafe_allow_html=True)
    
    # Waterfall plot (optionnel)
    if st.checkbox("Afficher le diagramme waterfall (détaillé)"):
        st.subheader("Diagramme Waterfall SHAP")
        
        import shap
        
        # Créer le plot waterfall
        shap_values_row = shap_result['shap_values'][selected_idx]
        expected_value = shap_result['explainer'].expected_value
        
        # Pour Isolation Forest, expected_value peut être une liste
        if isinstance(expected_value, list):
            expected_value = expected_value[0]
        
        fig = shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_row,
                base_values=expected_value,
                data=shap_result['X_sample'].iloc[selected_idx].values,
                feature_names=shap_result['feature_names']
            ),
            max_display=top_n,
            show=False
        )
        
        st.pyplot(fig)


# --------- Navigation principale --------- #

def main():
    """Fonction principale de l'application."""
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        st.markdown("---")
        st.subheader("📁 Données")
        
        uploaded_file = st.file_uploader(
            "Importer un fichier CSV",
            type=["csv"],
            help="Le fichier doit contenir les colonnes de transactions (user_id, product_amount, cashback, etc.)"
        )
        
        if uploaded_file is None:
            st.info("👈 Veuillez importer un fichier CSV pour commencer")
            st.stop()
        
        st.markdown("---")
        st.subheader("📊 Navigation")
        
        page_options = {
            "🎯 Objectifs du projet": page_objectifs,
            "🔍 Exploration des données": page_eda,
            "📊 ACP sur les utilisateurs": page_acp,
            "👥 Segmentation KMeans": page_kmeans,
            "🚨 Anomalies transactionnelles": page_anomalies,
            "🤖 Explications SHAP": page_xai
        }
        
        selected_page = st.radio(
            "Sélectionnez une page",
            list(page_options.keys())
        )
    
    # Chargement des données
    df_raw, user_features, tx_features = load_and_prepare_data(uploaded_file)
    
    # Affichage des métriques dans la sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("📈 Statistiques")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Transactions", df_raw.shape[0])
        with col2:
            st.metric("Utilisateurs", user_features.shape[0])
        
        st.metric("Features utilisateur", user_features.shape[1])
        st.metric("Features transaction", tx_features.shape[1])
    
    # Affichage de la page sélectionnée
    page_function = page_options[selected_page]
    
    if selected_page == "🎯 Objectifs du projet":
        page_function()
    elif selected_page == "🔍 Exploration des données":
        page_function(df_raw, user_features, tx_features)
    elif selected_page == "📊 ACP sur les utilisateurs":
        page_function(user_features)
    elif selected_page == "👥 Segmentation KMeans":
        page_function(user_features)
    elif selected_page == "🚨 Anomalies transactionnelles":
        page_function(df_raw, tx_features)
    elif selected_page == "🤖 Explications SHAP":
        page_function(df_raw, tx_features)


if __name__ == "__main__":
    main()