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

def page_acp(user_features: pd.DataFrame):
    """Page d'analyse PCA."""
    st.markdown('<h1 class="main-header">📊 Analyse en Composantes Principales</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Objectif de l'ACP</h3>
    <p>L'ACP permet de :</p>
    <ul>
    <li><b>Réduire la dimensionnalité</b> des données tout en conservant l'information</li>
    <li><b>Visualiser les individus</b> (utilisateurs) dans un espace réduit</li>
    <li><b>Analyser les relations</b> entre variables via les corrélations avec les axes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérifier si les features sont numériques
    non_numeric_cols = user_features.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if non_numeric_cols:
        st.warning(f"⚠️ Colonnes non-numériques détectées: {len(non_numeric_cols)}")
        with st.expander("Voir les colonnes non-numériques"):
            st.write(non_numeric_cols)
        
        st.info("""
        **Note :** L'ACP nécessite des données numériques. 
        Les colonnes non-numériques seront :
        1. Converties en variables numériques (one-hot encoding)
        2. Ou supprimées si la conversion n'est pas possible
        """)
    
    # Paramètres
    st.sidebar.subheader("Paramètres ACP")
    max_components = min(10, user_features.shape[1])
    n_components = st.sidebar.slider(
        "Nombre de composantes",
        min_value=2,
        max_value=max_components,
        value=min(3, max_components),
        help="Nombre de composantes principales à calculer"
    )
    
    # Bouton pour calculer PCA
    if st.button("🔧 Calculer l'ACP", type="primary"):
        try:
            with st.spinner("Calcul de l'ACP en cours..."):
                # Calcul PCA
                pca_result = compute_pca_cached(user_features, n_components)
                
                # Variance expliquée
                st.markdown('<h2 class="sub-header">Variance Expliquée</h2>', unsafe_allow_html=True)
                
                summary_df = get_pca_summary(pca_result)
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = make_subplots(rows=1, cols=2,
                                       subplot_titles=("Scree Plot", "Variance Cumulée"))
                    
                    # Scree plot
                    fig.add_trace(
                        go.Bar(x=summary_df['composante'], y=summary_df['variance_expliquee'],
                               name="Variance expliquée"),
                        row=1, col=1
                    )
                    
                    # Variance cumulée
                    fig.add_trace(
                        go.Scatter(x=summary_df['composante'], y=summary_df['variance_cumulee'],
                                  mode='lines+markers', name="Variance cumulée"),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(summary_df.style.format({
                        'variance_expliquee': '{:.3f}',
                        'variance_cumulee': '{:.3f}'
                    }), use_container_width=True)
                
                # ... [le reste du code ACP] ...
                
        except Exception as e:
            st.error(f"❌ Erreur lors du calcul de l'ACP: {str(e)}")
            st.info("""
            **Solution possible :**
            1. Vérifiez que vos données contiennent des colonnes numériques
            2. Essayez de réduire le nombre de composantes
            3. Vérifiez les types de données de vos colonnes
            """)

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
    st.markdown('<h1 class="main-header">🚨 Détection d\'Anomalies</h1>', unsafe_allow_html=True)
    
    # Vérifier les données
    st.info("Vérification des données transactionnelles...")
    
    # Afficher les types de données
    with st.expander("Afficher les types de données transactionnelles"):
        st.write(f"Shape: {tx_features.shape}")
        st.write(f"Types: {tx_features.dtypes.value_counts().to_dict()}")
    
    # Préparer les features transactionnelles
    try:
        # Filtrer uniquement les colonnes numériques
        tx_numeric = tx_features.select_dtypes(include=[np.number])
        
        if tx_numeric.empty:
            st.error("❌ Aucune colonne numérique trouvée dans les features transactionnelles!")
            st.info("""
            **Solution :**
            1. Vérifiez que vos données contiennent des colonnes numériques
            2. Les colonnes comme 'product_amount', 'cashback', etc. doivent être numériques
            """)
            return
        
        st.success(f"✅ {tx_numeric.shape[1]} colonnes numériques disponibles")
        
        # Paramètres
        contamination = st.slider(
            "Contamination (proportion d'anomalies attendue)",
            min_value=0.001,
            max_value=0.2,
            value=0.02,
            step=0.001,
            help="Proportion approximative d'anomalies dans les données"
        )
        
        if st.button("🔍 Détecter les anomalies", type="primary"):
            with st.spinner("Entraînement du modèle en cours..."):
                try:
                    # Entraînement
                    anomaly_result = train_isolation_forest(tx_numeric, contamination=contamination)
                    
                    # Statistiques
                    stats = get_anomaly_statistics(anomaly_result)
                    
                    # Afficher les résultats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Transactions analysées", stats['n_total'])
                    with col2:
                        st.metric("Anomalies détectées", stats['n_anomalies'])
                    with col3:
                        st.metric("Taux d'anomalies", f"{stats['pct_anomalies']:.1f}%")
                    with col4:
                        st.metric("Score moyen", f"{stats['score_mean']:.3f}")
                    
                    # ... reste du code pour afficher les anomalies ...
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la détection d'anomalies: {str(e)}")
                    st.info("""
                    **Solutions possibles :**
                    1. Réduisez le nombre de colonnes
                    2. Vérifiez qu'il n'y a pas de valeurs manquantes
                    3. Essayez avec contamination=0.05
                    """)
    
    except Exception as e:
        st.error(f"❌ Erreur dans la préparation des données: {str(e)}")

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