"""
Analyse en Composantes Principales (PCA) - Version Pro avec visualisations avancées
Version corrigée et robuste avec TOUTES les fonctions de visualisation
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CLASSE PRINCIPALE PCA
# ============================================================================

class PCAAnalyzer:
    """Classe complète pour l'analyse PCA avec visualisations avancées."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pca_result = None
        self.scaler = None
        self.pca = None
        logger.info("Initialisation de l'analyseur PCA")
    
    def fit(self, X: pd.DataFrame, n_components: Optional[int] = None, 
            variance_threshold: float = 0.9) -> Dict:
        """
        Effectue une PCA complète avec analyse des résultats.
        
        Args:
            X: DataFrame des données
            n_components: Nombre de composantes (None pour détermination automatique)
            variance_threshold: Seuil de variance cumulée pour détermination automatique
            
        Returns:
            Dictionnaire avec tous les résultats de l'analyse PCA
        """
        logger.info(f"🚀 Début de l'analyse PCA sur données de shape {X.shape}")
        
        try:
            # 1. Préparation des données
            logger.info("Étape 1/6: Préparation des données...")
            X_clean = self._prepare_data(X)
            
            if X_clean is None or X_clean.empty:
                raise ValueError("❌ Données vides après nettoyage")
            
            logger.info(f"✅ Données nettoyées: {X_clean.shape}")
            
            # 2. Standardisation robuste
            logger.info("Étape 2/6: Standardisation...")
            X_scaled, self.scaler, feature_names = self._safe_scale_features(X_clean)
            
            # VÉRIFICATION CRITIQUE
            if X_scaled is None:
                raise ValueError("❌ X_scaled est None après standardisation")
            if not isinstance(X_scaled, np.ndarray):
                X_scaled = np.array(X_scaled)
            if X_scaled.shape[1] == 0:
                raise ValueError("❌ Aucune colonne numérique après standardisation")
            
            logger.info(f"✅ Données standardisées: shape={X_scaled.shape}")
            
            # 3. Détermination du nombre optimal de composantes
            logger.info("Étape 3/6: Détermination des composantes...")
            if n_components is None:
                n_components = self._determine_optimal_components(
                    X_scaled, variance_threshold
                )
            else:
                n_components = min(n_components, X_scaled.shape[1])
            
            logger.info(f"📊 Nombre de composantes retenues: {n_components}")
            
            # 4. Calcul de la PCA
            logger.info("Étape 4/6: Calcul de la PCA...")
            try:
                self.pca = PCA(n_components=n_components, random_state=self.random_state)
                X_pca = self.pca.fit_transform(X_scaled)
                logger.info("✅ PCA calculée avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur dans PCA.fit_transform: {e}")
                # Fallback: PCA avec moins de composantes
                n_components = min(2, X_scaled.shape[1])
                self.pca = PCA(n_components=n_components, random_state=self.random_state)
                X_pca = self.pca.fit_transform(X_scaled)
                logger.warning(f"⚠ Fallback PCA avec {n_components} composantes")
            
            # 5. Calcul des métriques avancées
            logger.info("Étape 5/6: Calcul des métriques avancées...")
            metrics = self._compute_advanced_metrics(X_scaled, X_pca)
            
            # 6. Stockage des résultats
            logger.info("Étape 6/6: Construction des résultats...")
            self.pca_result = self._build_pca_result(
                X_clean, X_scaled, X_pca, feature_names, n_components, metrics
            )
            
            logger.info(f"✅ Analyse PCA terminée avec succès!")
            logger.info(f"   Composantes: {n_components}")
            logger.info(f"   Variance expliquée: {self.pca_result['cumulative_variance'][-1]:.3f}")
            
            return self.pca_result
            
        except Exception as e:
            logger.error(f"❌ Erreur dans l'analyse PCA: {e}", exc_info=True)
            raise
    
    # ... (toutes les méthodes de la classe restent identiques) ...

# ============================================================================
# FONCTIONS PUBLIQUES PRINCIPALES
# ============================================================================

def compute_pca(
    X: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.9,
    random_state: int = 42
) -> Dict:
    """
    Fonction wrapper principale pour la PCA.
    
    Args:
        X: DataFrame des données
        n_components: Nombre de composantes (None pour détermination automatique)
        variance_threshold: Seuil de variance cumulée
        random_state: Graine aléatoire
        
    Returns:
        Dictionnaire avec résultats PCA
    """
    logger.info("🚀 Démarrage de l'analyse PCA")
    
    try:
        analyzer = PCAAnalyzer(random_state=random_state)
        result = analyzer.fit(X, n_components, variance_threshold)
        logger.info("✅ Analyse PCA terminée avec succès")
        return result
    except Exception as e:
        logger.error(f"❌ Erreur fatale dans compute_pca: {e}", exc_info=True)
        raise ValueError(f"Impossible de calculer la PCA: {str(e)}")

# ============================================================================
# FONCTIONS DE VISUALISATION (INTÉGRÉES)
# ============================================================================

def create_scree_plot(pca_result: Dict) -> go.Figure:
    """Crée un scree plot amélioré avec coude marqué."""
    try:
        eigenvalues = pca_result['metrics']['eigenvalues']
        explained_var = pca_result['explained_variance_ratio'] * 100
        cum_var = pca_result['cumulative_variance'] * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Valeurs propres (Scree plot)", "Variance cumulée"),
            vertical_spacing=0.15
        )
        
        # Scree plot
        fig.add_trace(
            go.Bar(
                x=[f'PC{i+1}' for i in range(len(eigenvalues))],
                y=eigenvalues,
                name='Valeur propre',
                marker_color='royalblue',
                text=[f'{ev:.2f}' for ev in eigenvalues],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Ligne horizontale à 1 (critère de Kaiser)
        fig.add_hline(
            y=1, line_dash="dash", line_color="red",
            annotation_text="Critère de Kaiser", 
            annotation_position="top right",
            row=1, col=1
        )
        
        # Variance cumulée
        fig.add_trace(
            go.Scatter(
                x=[f'PC{i+1}' for i in range(len(cum_var))],
                y=cum_var,
                mode='lines+markers+text',
                name='Variance cumulée',
                line=dict(color='firebrick', width=3),
                marker=dict(size=10),
                text=[f'{v:.1f}%' for v in cum_var],
                textposition="top center"
            ),
            row=2, col=1
        )
        
        # Lignes horizontales pour les seuils
        thresholds = [70, 80, 90, 95]
        colors = ['lightgray', 'gray', 'darkgray', 'black']
        
        for thresh, color in zip(thresholds, colors):
            fig.add_hline(
                y=thresh, line_dash="dot", line_color=color,
                annotation_text=f'{thresh}%', 
                annotation_position="right",
                row=2, col=1
            )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Analyse des valeurs propres et variance expliquée",
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Composantes principales", row=1, col=1)
        fig.update_yaxes(title_text="Valeur propre", row=1, col=1)
        fig.update_xaxes(title_text="Composantes principales", row=2, col=1)
        fig.update_yaxes(title_text="Variance cumulée (%)", row=2, col=1)
        
        return fig
    except Exception as e:
        logger.error(f"Erreur dans create_scree_plot: {e}")
        return go.Figure()

def create_correlation_circle(pca_result: Dict, pc_x: int = 0, pc_y: int = 1, 
                            n_variables: int = 20, threshold: float = 0.3) -> go.Figure:
    """Crée un cercle des corrélations avancé."""
    try:
        loadings = pca_result['loadings']
        feature_names = pca_result['feature_names']
        cos2 = pca_result['metrics']['variable_cos2']
        
        # S'assurer que les indices sont valides
        pc_x = min(pc_x, pca_result['n_components'] - 1)
        pc_y = min(pc_y, pca_result['n_components'] - 1)
        
        # Préparer les données
        df_vars = pd.DataFrame({
            'variable': feature_names,
            'cor_x': loadings[:, pc_x],
            'cor_y': loadings[:, pc_y],
            'cos2': cos2,
            'length': np.sqrt(loadings[:, pc_x]**2 + loadings[:, pc_y]**2)
        })
        
        # Filtrer par qualité de représentation
        df_vars = df_vars[df_vars['length'] >= threshold]
        
        # Limiter le nombre de variables
        if len(df_vars) > n_variables:
            df_vars = df_vars.nlargest(n_variables, 'length')
        
        # Créer la figure
        fig = go.Figure()
        
        # Cercle unité
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines', line=dict(color='lightgray', dash='dash'),
            name='Cercle unité'
        ))
        
        # Axes
        fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0,
                      line=dict(color="gray", width=1))
        fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1,
                      line=dict(color="gray", width=1))
        
        # Variables
        if len(df_vars) > 0:
            fig.add_trace(go.Scatter(
                x=df_vars['cor_x'], y=df_vars['cor_y'],
                mode='markers+text',
                marker=dict(
                    size=df_vars['cos2'] * 30 + 10,  # Taille proportionnelle à cos2
                    color=df_vars['length'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Qualité")
                ),
                text=df_vars['variable'],
                textposition="top center",
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Corrélation PC1: %{x:.3f}<br>" +
                    "Corrélation PC2: %{y:.3f}<br>" +
                    "Qualité: %{marker.color:.3f}<br>" +
                    "Cos2: %{customdata[0]:.3f}<extra></extra>"
                ),
                customdata=np.stack((df_vars['cos2'],), axis=-1)
            ))
            
            # Vecteurs
            for _, row in df_vars.iterrows():
                fig.add_trace(go.Scatter(
                    x=[0, row['cor_x']], y=[0, row['cor_y']],
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title=f"Cercle des corrélations - PC{pc_x+1} vs PC{pc_y+1}",
            xaxis_title=f"PC{pc_x+1} ({pca_result['explained_variance_ratio'][pc_x]*100:.1f}%)",
            yaxis_title=f"PC{pc_y+1} ({pca_result['explained_variance_ratio'][pc_y]*100:.1f}%)",
            xaxis=dict(range=[-1.1, 1.1], zeroline=True),
            yaxis=dict(range=[-1.1, 1.1], zeroline=True),
            hovermode='closest',
            template="plotly_white",
            height=600
        )
        
        return fig
    except Exception as e:
        logger.error(f"Erreur dans create_correlation_circle: {e}")
        return go.Figure()

def create_biplot(pca_result: Dict, pc_x: int = 0, pc_y: int = 1,
                 n_individuals: int = 100, n_variables: int = 10) -> go.Figure:
    """Crée un biplot (individus + variables)."""
    try:
        X_pca = pca_result['X_pca']
        loadings = pca_result['loadings']
        feature_names = pca_result['feature_names']
        
        # S'assurer que les indices sont valides
        pc_x = min(pc_x, pca_result['n_components'] - 1)
        pc_y = min(pc_y, pca_result['n_components'] - 1)
        
        # Échantillonner les individus
        if len(X_pca) > n_individuals:
            idx = np.random.choice(len(X_pca), n_individuals, replace=False)
            individuals = X_pca[idx]
        else:
            individuals = X_pca
        
        # Sélectionner les variables importantes
        importance = np.sqrt(loadings[:, pc_x]**2 + loadings[:, pc_y]**2)
        top_vars_idx = np.argsort(importance)[-n_variables:]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Plan factoriel", "Zoom sur les variables"),
            column_widths=[0.7, 0.3]
        )
        
        # Individus
        fig.add_trace(
            go.Scatter(
                x=individuals[:, pc_x], y=individuals[:, pc_y],
                mode='markers',
                marker=dict(size=8, color='lightblue', opacity=0.6),
                name='Individus',
                hovertemplate="Individu %{customdata}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                customdata=np.arange(len(individuals))
            ),
            row=1, col=1
        )
        
        # Variables (dans le grand graphique)
        if len(top_vars_idx) > 0:
            scale_factor = 0.8 * np.max(np.abs(individuals)) / np.max(np.abs(loadings[top_vars_idx]))
            
            for i in top_vars_idx:
                fig.add_trace(
                    go.Scatter(
                        x=[0, loadings[i, pc_x] * scale_factor],
                        y=[0, loadings[i, pc_y] * scale_factor],
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[loadings[i, pc_x] * scale_factor],
                        y=[loadings[i, pc_y] * scale_factor],
                        mode='markers+text',
                        marker=dict(color='red', size=10),
                        text=[feature_names[i]],
                        textposition="top center",
                        name=feature_names[i],
                        showlegend=False,
                        hovertemplate=f"<b>{feature_names[i]}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
                    ),
                    row=1, col=1
                )
        
        # Cercle des corrélations (zoom)
        try:
            circle_fig = create_correlation_circle(pca_result, pc_x, pc_y, n_variables)
            if len(circle_fig.data) > 0:
                fig.add_trace(circle_fig.data[0], row=1, col=2)
        except:
            pass
        
        fig.update_layout(
            title=f"Biplot - PC{pc_x+1} vs PC{pc_y+1}",
            height=600,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text=f"PC{pc_x+1}", row=1, col=1)
        fig.update_yaxes(title_text=f"PC{pc_y+1}", row=1, col=1)
        fig.update_xaxes(title_text=f"PC{pc_x+1}", range=[-1.1, 1.1], row=1, col=2)
        fig.update_yaxes(title_text=f"PC{pc_y+1}", range=[-1.1, 1.1], row=1, col=2)
        
        return fig
    except Exception as e:
        logger.error(f"Erreur dans create_biplot: {e}")
        return go.Figure()

def create_3d_pca_plot(pca_result: Dict) -> go.Figure:
    """Crée une visualisation 3D des trois premières composantes."""
    try:
        if pca_result['n_components'] < 3:
            raise ValueError("Au moins 3 composantes sont nécessaires pour la visualisation 3D")
        
        X_pca = pca_result['X_pca']
        
        # Qualité de représentation pour la couleur
        cos2 = pca_result['metrics']['cos2_individuals']
        
        fig = go.Figure(data=go.Scatter3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=cos2,
                colorscale='Viridis',
                colorbar=dict(title="Qualité (cos2)"),
                showscale=True
            ),
            text=[f"Individu {i}" for i in range(len(X_pca))],
            hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br>Qualité: %{marker.color:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Visualisation 3D des individus sur les trois premières composantes",
            scene=dict(
                xaxis_title=f"PC1 ({pca_result['explained_variance_ratio'][0]*100:.1f}%)",
                yaxis_title=f"PC2 ({pca_result['explained_variance_ratio'][1]*100:.1f}%)",
                zaxis_title=f"PC3 ({pca_result['explained_variance_ratio'][2]*100:.1f}%)"
            ),
            height=700,
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Erreur dans create_3d_pca_plot: {e}")
        return go.Figure()

# ============================================================================
# FONCTIONS UTILITAIRES (autres fonctions nécessaires)
# ============================================================================

def get_pca_summary(pca_result: Dict) -> pd.DataFrame:
    """Crée un DataFrame récapitulatif de la PCA."""
    try:
        n_components = pca_result['n_components']
        
        summary = pd.DataFrame({
            'composante': [f'PC{i+1}' for i in range(n_components)],
            'variance_expliquee': pca_result['explained_variance_ratio'],
            'variance_cumulee': pca_result['cumulative_variance'],
            'variance_pct': pca_result['explained_variance_ratio'] * 100,
            'variance_cumulee_pct': np.cumsum(pca_result['explained_variance_ratio']) * 100,
            'eigenvalue': pca_result['metrics']['eigenvalues'][:n_components],
            'inertia': [pca_result['metrics']['inertia']] * n_components
        })
        
        return summary.round(4)
    except Exception as e:
        logger.error(f"Erreur dans get_pca_summary: {e}")
        return pd.DataFrame({'error': [str(e)]})

def get_top_loadings(
    pca_result: Dict, 
    component: int = 0, 
    n_features: int = 10,
    threshold: float = 0.3
) -> pd.DataFrame:
    """Récupère les variables les plus corrélées avec une composante."""
    try:
        if component >= pca_result['n_components']:
            raise ValueError(f"Component {component} n'existe pas. Maximum: {pca_result['n_components']-1}")
        
        loadings = pca_result['loadings'][:, component]
        feature_names = pca_result['feature_names']
        
        # Qualité de représentation (cos2)
        cos2 = pca_result['metrics']['variable_cos2']
        
        df_loadings = pd.DataFrame({
            'variable': feature_names,
            'loading': loadings,
            'abs_loading': np.abs(loadings),
            'cos2': cos2,
            'contribution': (loadings**2) * 100 / np.sum(loadings**2) if np.sum(loadings**2) > 0 else 0
        })
        
        # Filtrer par seuil
        df_loadings = df_loadings[df_loadings['abs_loading'] >= threshold]
        
        # Trier par valeur absolue
        df_loadings = df_loadings.sort_values('abs_loading', ascending=False).head(n_features)
        
        # Catégoriser l'importance
        try:
            df_loadings['importance'] = pd.cut(
                df_loadings['abs_loading'],
                bins=[0, 0.3, 0.6, 0.8, 1],
                labels=['Faible', 'Moyenne', 'Forte', 'Très forte']
            )
        except:
            df_loadings['importance'] = 'Non classé'
        
        return df_loadings.round(4)
    except Exception as e:
        logger.error(f"Erreur dans get_top_loadings: {e}")
        return pd.DataFrame({'error': [str(e)]})

def suggest_optimal_components(pca_result: Dict, thresholds: List[float] = None) -> pd.DataFrame:
    """Suggère des nombres optimaux de composantes avec différentes méthodes."""
    try:
        if thresholds is None:
            thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        cumulative_variance = pca_result['cumulative_variance']
        eigenvalues = pca_result['metrics']['eigenvalues']
        
        suggestions = []
        
        # Méthode 1: Variance cumulée
        for threshold in thresholds:
            try:
                n_components = np.argmax(cumulative_variance >= threshold) + 1
                if n_components == 0:  # Si aucune composante n'atteint le seuil
                    n_components = len(cumulative_variance)
                suggestions.append({
                    'méthode': f'Variance {int(threshold*100)}%',
                    'n_composantes': n_components,
                    'variance_expliquée': cumulative_variance[n_components-1] if n_components > 0 else 0,
                    'description': f'Explique {threshold*100:.0f}% de la variance'
                })
            except:
                continue
        
        # Méthode 2: Kaiser (eigenvalue > 1)
        try:
            n_kaiser = np.sum(eigenvalues > 1)
            suggestions.append({
                'méthode': 'Critère de Kaiser',
                'n_composantes': n_kaiser,
                'variance_expliquée': cumulative_variance[n_kaiser-1] if n_kaiser > 0 else 0,
                'description': 'Valeurs propres > 1'
            })
        except:
            pass
        
        return pd.DataFrame(suggestions)
    except Exception as e:
        logger.error(f"Erreur dans suggest_optimal_components: {e}")
        return pd.DataFrame({'error': [str(e)]})

def generate_pca_report(pca_result: Dict) -> str:
    """Génère un rapport textuel complet de l'analyse PCA."""
    try:
        n_components = pca_result['n_components']
        total_variance = pca_result['cumulative_variance'][-1]
        metrics = pca_result['metrics']
        
        report = f"""
==============================================
RAPPORT D'ANALYSE EN COMPOSANTES PRINCIPALES
==============================================

I. SYNTHÈSE GÉNÉRALE
--------------------
- Nombre d'individus : {len(pca_result['X_pca'])}
- Nombre de variables initiales : {len(pca_result['feature_names'])}
- Nombre de composantes retenues : {n_components}
- Variance totale expliquée : {total_variance:.1%}

II. QUALITÉ DE L'ANALYSE
------------------------
- Test de sphéricité de Bartlett : {'Significatif' if metrics.get('bartlett_sphericity', {}).get('significant', False) else 'Non significatif'}
- Indice KMO : {metrics.get('kmo_measure', {}).get('kmo', 0):.3f} ({metrics.get('kmo_measure', {}).get('interpretation', 'Non calculé')})
- Qualité moyenne de représentation : {metrics['mean_cos2']:.1%}

III. COMPOSANTES PRINCIPALES
---------------------------
"""
        
        for i in range(min(3, n_components)):
            var_exp = pca_result['explained_variance_ratio'][i] * 100
            cum_var = pca_result['cumulative_variance'][i] * 100
            
            report += f"""
Composante PC{i+1} :
- Variance expliquée : {var_exp:.1f}%
- Variance cumulée : {cum_var:.1f}%
- Valeur propre : {metrics['eigenvalues'][i]:.3f}
"""
        
        # Top variables
        report += """

IV. INTERPRÉTATION
------------------
"""
        
        if total_variance > 0.8:
            report += "✓ Excellente réduction de dimensionnalité\n"
        elif total_variance > 0.6:
            report += "✓ Bonne réduction de dimensionnalité\n"
        else:
            report += "⚠ Réduction de dimensionnalité modérée\n"
        
        if metrics.get('kmo_measure', {}).get('adequate', False):
            report += "✓ Les données sont adéquates pour l'analyse factorielle\n"
        else:
            report += "⚠ Les données sont peu adaptées à l'analyse factorielle\n"
        
        report += "\n" + "="*50 + "\n"
        
        return report
        
    except Exception as e:
        logger.error(f"Erreur dans generate_pca_report: {e}")
        return f"❌ Erreur dans la génération du rapport: {str(e)}"

# ============================================================================
# VERSION SIMPLIFIÉE POUR VOTRE APPLICATION STREAMLIT
# ============================================================================

def page_acp_simplified(user_features: pd.DataFrame):
    """Version simplifiée de la page ACP pour votre application Streamlit."""
    import streamlit as st
    
    st.markdown('<h1 class="main-header">📊 Analyse en Composantes Principales</h1>', unsafe_allow_html=True)
    
    # Paramètres
    with st.sidebar.expander("⚙️ Paramètres PCA", expanded=False):
        n_components = st.slider(
            "Nombre de composantes",
            min_value=2,
            max_value=min(10, user_features.shape[1]),
            value=min(3, user_features.shape[1]),
            help="Nombre de composantes principales à calculer"
        )
        
        variance_threshold = st.slider(
            "Seuil de variance minimale",
            min_value=0.5,
            max_value=0.99,
            value=0.9,
            step=0.05,
            help="Variance minimale à conserver si choix automatique"
        )
    
    # Bouton de calcul
    if st.button("🚀 Lancer l'analyse PCA", type="primary", use_container_width=True):
        with st.spinner("Analyse PCA en cours..."):
            try:
                # Calcul PCA
                pca_result = compute_pca(
                    user_features, 
                    n_components=n_components,
                    variance_threshold=variance_threshold
                )
                
                # Métriques
                st.markdown("### 📈 Métriques principales")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_var = pca_result['cumulative_variance'][-1] * 100
                    st.metric("Variance totale expliquée", f"{total_var:.1f}%")
                
                with col2:
                    avg_quality = pca_result['metrics']['mean_cos2'] * 100
                    st.metric("Qualité moyenne", f"{avg_quality:.1f}%")
                
                with col3:
                    kmo = pca_result['metrics'].get('kmo_measure', {}).get('kmo', 0)
                    st.metric("Indice KMO", f"{kmo:.3f}")
                
                with col4:
                    st.metric("Composantes retenues", pca_result['n_components'])
                
                # Visualisations
                tab1, tab2, tab3 = st.tabs(["📊 Scree Plot", "🔵 Cercle des corrélations", "🎯 Biplot"])
                
                with tab1:
                    st.plotly_chart(create_scree_plot(pca_result), use_container_width=True)
                
                with tab2:
                    pc_x = st.selectbox("Composante X", range(pca_result['n_components']), 0, key="corr_x")
                    pc_y = st.selectbox("Composante Y", range(pca_result['n_components']), 1, 
                                      key="corr_y", disabled=pc_x)
                    
                    st.plotly_chart(
                        create_correlation_circle(pca_result, pc_x, pc_y), 
                        use_container_width=True
                    )
                
                with tab3:
                    if pca_result['n_components'] >= 2:
                        st.plotly_chart(
                            create_biplot(pca_result), 
                            use_container_width=True
                        )
                
                # Visualisation 3D
                if pca_result['n_components'] >= 3:
                    st.subheader("🌐 Visualisation 3D")
                    st.plotly_chart(create_3d_pca_plot(pca_result), use_container_width=True)
                
                # Rapport
                with st.expander("📄 Voir le rapport complet"):
                    report = generate_pca_report(pca_result)
                    st.code(report)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse PCA: {str(e)}")
    else:
        st.info("👈 Cliquez sur le bouton ci-dessus pour lancer l'analyse PCA")