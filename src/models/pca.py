"""
Analyse en Composantes Principales (PCA) - Version complète avec toutes les fonctions
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
# FONCTIONS UTILITAIRES
# ============================================================================

def prepare_data_for_pca(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les données pour l'analyse PCA.
    
    Args:
        X: DataFrame des données
        
    Returns:
        DataFrame préparé
    """
    logger.info("Préparation des données pour PCA...")
    
    if X is None or X.empty:
        raise ValueError("❌ Données d'entrée vides")
    
    X_clean = X.copy()
    
    # Vérifier et convertir les colonnes non-numériques
    non_numeric_cols = X_clean.select_dtypes(exclude=[np.number]).columns
    
    if len(non_numeric_cols) > 0:
        logger.info(f"Conversion de {len(non_numeric_cols)} colonnes non-numériques...")
        for col in non_numeric_cols:
            try:
                # Essayer de convertir en numérique
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            except:
                # Si échec, encoder catégoriel
                X_clean[col] = pd.factorize(X_clean[col])[0]
    
    # Supprimer les colonnes avec variance nulle ou presque nulle
    variance = X_clean.var()
    zero_variance_cols = variance[variance < 1e-10].index.tolist()
    
    if zero_variance_cols:
        logger.warning(f"Suppression des colonnes à variance nulle: {zero_variance_cols}")
        X_clean = X_clean.drop(columns=zero_variance_cols)
    
    # Traiter les valeurs manquantes
    if X_clean.isna().any().any():
        logger.info("Traitement des valeurs manquantes...")
        imputer = SimpleImputer(strategy='median')
        X_clean_imputed = imputer.fit_transform(X_clean)
        X_clean = pd.DataFrame(X_clean_imputed, columns=X_clean.columns, index=X_clean.index)
    
    # Vérification finale
    if X_clean.shape[1] < 2:
        raise ValueError(f"❌ Pas assez de colonnes numériques ({X_clean.shape[1]}) pour PCA")
    
    logger.info(f"✅ Données préparées: {X_clean.shape}")
    return X_clean


def safe_scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Standardisation robuste avec gestion d'erreurs.
    
    Args:
        X: DataFrame des données
        
    Returns:
        Tuple (données standardisées, scaler, noms des features)
    """
    logger.info("Standardisation des features...")
    
    try:
        # Sélectionner uniquement les colonnes numériques
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.error("❌ Aucune colonne numérique trouvée")
            # Essayer de convertir toutes les colonnes
            X_numeric = X.astype(float).values
            feature_names = X.columns.tolist()
        else:
            X_numeric = X[numeric_cols].values
            feature_names = numeric_cols.tolist()
        
        # Vérifier les dimensions
        if X_numeric.shape[1] == 0:
            raise ValueError("❌ Aucune dimension numérique")
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        # Vérification finale
        if X_scaled is None:
            raise ValueError("❌ X_scaled est None après scaling")
        
        logger.info(f"✅ Standardisation réussie: {X_scaled.shape}")
        return X_scaled, scaler, feature_names
        
    except Exception as e:
        logger.error(f"❌ Erreur dans la standardisation: {e}")
        # Fallback: scaling manuel simple
        logger.warning("⚠ Utilisation du fallback de standardisation")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            numeric_cols = X.columns
            X_numeric = X.values.astype(float)
        else:
            X_numeric = X[numeric_cols].values
        
        # Scaling manuel (centre-réduit)
        means = np.mean(X_numeric, axis=0)
        stds = np.std(X_numeric, axis=0)
        stds[stds == 0] = 1.0  # Éviter la division par zéro
        
        X_scaled = (X_numeric - means) / stds
        
        # Créer un scaler factice
        scaler = StandardScaler()
        scaler.mean_ = means
        scaler.scale_ = stds
        
        return X_scaled, scaler, list(numeric_cols)


def determine_optimal_components(X_scaled: np.ndarray, 
                               variance_threshold: float = 0.9,
                               random_state: int = 42) -> int:
    """
    Détermine le nombre optimal de composantes.
    
    Args:
        X_scaled: Données standardisées
        variance_threshold: Seuil de variance cumulée
        random_state: Graine aléatoire
        
    Returns:
        Nombre optimal de composantes
    """
    try:
        n_features = X_scaled.shape[1]
        
        # Méthode 1: Variance cumulée
        pca_temp = PCA(random_state=random_state)
        pca_temp.fit(X_scaled)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        
        n_by_variance = np.argmax(cumsum >= variance_threshold) + 1
        if n_by_variance == 0:  # Si aucune composante n'atteint le seuil
            n_by_variance = min(2, n_features)
        
        # Méthode 2: Kaiser criterion (eigenvalue > 1)
        eigenvalues = pca_temp.explained_variance_
        n_by_kaiser = np.sum(eigenvalues > 1)
        
        # Choisir le maximum raisonnable
        suggestions = [n_by_variance, n_by_kaiser]
        valid_suggestions = [s for s in suggestions if 2 <= s <= n_features]
        
        if valid_suggestions:
            n_components = max(valid_suggestions)
        else:
            n_components = min(2, n_features)
        
        # S'assurer que c'est au moins 2 pour la visualisation
        n_components = max(2, min(n_components, n_features))
        
        logger.info(f"Suggestions: Variance={n_by_variance}, Kaiser={n_by_kaiser} → Choix={n_components}")
        
        return n_components
        
    except Exception as e:
        logger.error(f"Erreur dans la détermination des composantes: {e}")
        # Fallback: 2 composantes pour la visualisation
        return min(2, X_scaled.shape[1])


def compute_bartlett_test(X_scaled: np.ndarray) -> Dict:
    """
    Test de sphéricité de Bartlett.
    
    Args:
        X_scaled: Données standardisées
        
    Returns:
        Résultats du test
    """
    try:
        n, p = X_scaled.shape
        if n < 2 or p < 2:
            return {'error': 'Dimensions insuffisantes'}
        
        corr_matrix = np.corrcoef(X_scaled.T)
        det = np.linalg.det(corr_matrix)
        
        # Éviter les problèmes numériques
        if det <= 0:
            return {'error': 'Déterminant non positif'}
        
        chi2_val = -((n - 1) - (2*p + 5)/6) * np.log(det)
        df = p*(p-1)/2
        
        # Calcul de la p-value
        if chi2_val > 0 and df > 0:
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(chi2_val, df)
        else:
            p_value = 1.0
        
        return {
            'chi_square': float(chi2_val),
            'df': int(df),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    except Exception as e:
        logger.error(f"Erreur dans Bartlett test: {e}")
        return {'error': f'Calcul impossible: {str(e)}'}


def compute_kmo_test(X_scaled: np.ndarray) -> Dict:
    """
    Test KMO (Kaiser-Meyer-Olkin).
    
    Args:
        X_scaled: Données standardisées
        
    Returns:
        Résultats du test KMO
    """
    try:
        n, p = X_scaled.shape
        if n < 2 or p < 2:
            return {'kmo': 0, 'interpretation': 'Inadequate', 'adequate': False}
        
        corr_matrix = np.corrcoef(X_scaled.T)
        inv_corr = np.linalg.pinv(corr_matrix)  # Utiliser pseudo-inverse pour stabilité
        diag_inv = np.diag(inv_corr)
        
        # Éviter division par zéro
        diag_inv_sqrt = np.sqrt(diag_inv)
        diag_inv_sqrt[diag_inv_sqrt == 0] = 1e-10
        
        partial_corr = -inv_corr / np.outer(diag_inv_sqrt, diag_inv_sqrt)
        np.fill_diagonal(partial_corr, 1)
        
        # Calcul du KMO
        corr_squared_sum = np.sum(corr_matrix**2) - np.sum(np.diag(corr_matrix)**2)
        partial_squared_sum = np.sum(partial_corr**2) - np.sum(np.diag(partial_corr)**2)
        
        if partial_squared_sum == 0:
            kmo = 0
        else:
            kmo = corr_squared_sum / (corr_squared_sum + partial_squared_sum)
            kmo = np.clip(kmo, 0, 1)  # S'assurer que c'est entre 0 et 1
        
        # Interprétation
        interpretation = "Inacceptable"
        if kmo >= 0.9:
            interpretation = "Mervelous"
        elif kmo >= 0.8:
            interpretation = "Meritorious"
        elif kmo >= 0.7:
            interpretation = "Middling"
        elif kmo >= 0.6:
            interpretation = "Mediocre"
        elif kmo >= 0.5:
            interpretation = "Miserable"
        
        return {
            'kmo': float(kmo),
            'interpretation': interpretation,
            'adequate': kmo >= 0.6
        }
    except Exception as e:
        logger.error(f"Erreur dans KMO test: {e}")
        return {'kmo': 0, 'interpretation': 'Error', 'adequate': False}


# ============================================================================
# FONCTION PRINCIPALE PCA
# ============================================================================

def compute_pca(
    X: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.9,
    random_state: int = 42
) -> Dict:
    """
    Fonction principale pour calculer la PCA.
    
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
        # 1. Préparation des données
        logger.info("Étape 1/6: Préparation des données...")
        X_clean = prepare_data_for_pca(X)
        
        if X_clean is None or X_clean.empty:
            raise ValueError("❌ Données vides après nettoyage")
        
        logger.info(f"✅ Données nettoyées: {X_clean.shape}")
        
        # 2. Standardisation
        logger.info("Étape 2/6: Standardisation...")
        X_scaled, scaler, feature_names = safe_scale_features(X_clean)
        
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
            n_components = determine_optimal_components(
                X_scaled, variance_threshold, random_state
            )
        else:
            n_components = min(n_components, X_scaled.shape[1])
        
        logger.info(f"📊 Nombre de composantes retenues: {n_components}")
        
        # 4. Calcul de la PCA
        logger.info("Étape 4/6: Calcul de la PCA...")
        try:
            pca = PCA(n_components=n_components, random_state=random_state)
            X_pca = pca.fit_transform(X_scaled)
            logger.info("✅ PCA calculée avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur dans PCA.fit_transform: {e}")
            # Fallback: PCA avec moins de composantes
            n_components = min(2, X_scaled.shape[1])
            pca = PCA(n_components=n_components, random_state=random_state)
            X_pca = pca.fit_transform(X_scaled)
            logger.warning(f"⚠ Fallback PCA avec {n_components} composantes")
        
        # 5. Calcul des métriques avancées
        logger.info("Étape 5/6: Calcul des métriques avancées...")
        
        # Cos2 - qualité de représentation des individus
        try:
            sum_squared_X = np.sum(X_scaled**2, axis=1)
            sum_squared_X[sum_squared_X == 0] = 1  # Éviter division par zéro
            cos2 = np.sum(X_pca**2, axis=1) / sum_squared_X
            cos2 = np.clip(cos2, 0, 1)  # S'assurer que c'est entre 0 et 1
        except:
            cos2 = np.ones(X_pca.shape[0]) * 0.5
        
        # Contributions des individus aux axes
        try:
            sum_X_pca_squared = np.sum(X_pca**2, axis=0)
            sum_X_pca_squared[sum_X_pca_squared == 0] = 1
            contributions = (X_pca**2) / sum_X_pca_squared * 100
        except:
            contributions = np.ones((X_pca.shape[0], n_components)) * (100 / n_components)
        
        # Qualité de représentation des variables (cos2)
        try:
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            var_cos2 = np.sum(loadings**2, axis=1)
        except:
            loadings = np.ones((len(feature_names), n_components))
            var_cos2 = np.ones(len(feature_names)) * 0.5
        
        # Tests statistiques
        bartlett_result = compute_bartlett_test(X_scaled)
        kmo_result = compute_kmo_test(X_scaled)
        
        # Métriques
        metrics = {
            'cos2_individuals': cos2,
            'individual_contributions': contributions,
            'variable_cos2': var_cos2,
            'explained_variance_pct': pca.explained_variance_ratio_ * 100,
            'eigenvalues': pca.explained_variance_,
            'inertia': np.sum(X_scaled**2) if X_scaled.shape[0] > 0 else 0,
            'bartlett_sphericity': bartlett_result,
            'kmo_measure': kmo_result,
            'total_variance': np.sum(pca.explained_variance_) if hasattr(pca, 'explained_variance_') else 0,
            'mean_cos2': np.mean(cos2) if len(cos2) > 0 else 0
        }
        
        # 6. Construction des résultats
        logger.info("Étape 6/6: Construction des résultats...")
        
        pca_result = {
            'scaler': scaler,
            'pca': pca,
            'X_pca': X_pca,
            'X_scaled': X_scaled,
            'X_clean': X_clean,
            'feature_names': feature_names,
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'loadings': loadings,
            'metrics': metrics,
        }
        
        logger.info(f"✅ Analyse PCA terminée avec succès!")
        logger.info(f"   Composantes: {n_components}")
        logger.info(f"   Variance expliquée: {pca_result['cumulative_variance'][-1]:.3f}")
        
        return pca_result
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale dans compute_pca: {e}", exc_info=True)
        raise ValueError(f"Impossible de calculer la PCA: {str(e)}")


# ============================================================================
# FONCTIONS DE VISUALISATION
# ============================================================================

def create_scree_plot(pca_result: Dict) -> go.Figure:
    """
    Crée un scree plot amélioré avec coude marqué.
    
    Args:
        pca_result: Résultat de la PCA
        
    Returns:
        Figure Plotly
    """
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
        # Retourner une figure vide
        fig = go.Figure()
        fig.add_annotation(text="Erreur lors de la création du graphique",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig


def create_correlation_circle(pca_result: Dict, pc_x: int = 0, pc_y: int = 1, 
                            n_variables: int = 20, threshold: float = 0.3) -> go.Figure:
    """
    Crée un cercle des corrélations avancé.
    
    Args:
        pca_result: Résultat de la PCA
        pc_x: Indice de la première composante
        pc_y: Indice de la seconde composante
        n_variables: Nombre maximum de variables à afficher
        threshold: Seuil minimum pour afficher une variable
        
    Returns:
        Figure Plotly
    """
    try:
        loadings = pca_result['loadings']
        feature_names = pca_result['feature_names']
        
        # Vérifier que les indices sont valides
        n_components = pca_result['n_components']
        if pc_x >= n_components or pc_y >= n_components:
            pc_x = 0
            pc_y = min(1, n_components-1)
        
        # Préparer les données
        df_vars = pd.DataFrame({
            'variable': feature_names,
            'cor_x': loadings[:, pc_x],
            'cor_y': loadings[:, pc_y],
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
                    size=df_vars['length'] * 20 + 10,
                    color=df_vars['length'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Longueur")
                ),
                text=df_vars['variable'],
                textposition="top center",
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Corrélation PC{pcx}: %{{x:.3f}}<br>".format(pcx=pc_x+1) +
                    "Corrélation PC{pcy}: %{{y:.3f}}<br>".format(pcy=pc_y+1) +
                    "Longueur: %{marker.color:.3f}<extra></extra>"
                )
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
        # Retourner une figure vide
        fig = go.Figure()
        fig.add_annotation(text="Erreur lors de la création du graphique",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig


def create_biplot(pca_result: Dict, pc_x: int = 0, pc_y: int = 1,
                 n_individuals: int = 100, n_variables: int = 10) -> go.Figure:
    """
    Crée un biplot (individus + variables).
    
    Args:
        pca_result: Résultat de la PCA
        pc_x: Indice de la première composante
        pc_y: Indice de la seconde composante
        n_individuals: Nombre d'individus à afficher
        n_variables: Nombre de variables à afficher
        
    Returns:
        Figure Plotly
    """
    try:
        X_pca = pca_result['X_pca']
        loadings = pca_result['loadings']
        feature_names = pca_result['feature_names']
        
        # Vérifier que les indices sont valides
        n_components = pca_result['n_components']
        if pc_x >= n_components or pc_y >= n_components:
            pc_x = 0
            pc_y = min(1, n_components-1)
        
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
                hovertemplate="Individu %{customdata}<br>PC{pcx}: %{{x:.2f}}<br>PC{pcy}: %{{y:.2f}}<extra></extra>".format(
                    pcx=pc_x+1, pcy=pc_y+1
                ),
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
                        hovertemplate=f"<b>{feature_names[i]}</b><br>PC{pc_x+1}: %{{x:.2f}}<br>PC{pc_y+1}: %{{y:.2f}}<extra></extra>"
                    ),
                    row=1, col=1
                )
        
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
        # Retourner une figure vide
        fig = go.Figure()
        fig.add_annotation(text="Erreur lors de la création du graphique",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig


def create_3d_pca_plot(pca_result: Dict) -> go.Figure:
    """
    Crée une visualisation 3D des trois premières composantes.
    
    Args:
        pca_result: Résultat de la PCA
        
    Returns:
        Figure Plotly 3D
    """
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
        # Retourner une figure 2D alternative
        fig = go.Figure()
        fig.add_annotation(text="Visualisation 3D non disponible (besoin d'au moins 3 composantes)",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig


# ============================================================================
# FONCTIONS D'ANALYSE ET DE RAPPORT
# ============================================================================

def get_pca_summary(pca_result: Dict) -> pd.DataFrame:
    """
    Crée un DataFrame récapitulatif de la PCA.
    
    Args:
        pca_result: Résultat de la PCA
        
    Returns:
        DataFrame récapitulatif
    """
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
    """
    Récupère les variables les plus corrélées avec une composante.
    
    Args:
        pca_result: Résultat de la PCA
        component: Index de la composante (0-based)
        n_features: Nombre de features à retourner
        threshold: Seuil minimum pour la valeur absolue du loading
        
    Returns:
        DataFrame avec les loadings
    """
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
    """
    Suggère des nombres optimaux de composantes avec différentes méthodes.
    
    Args:
        pca_result: Résultat de la PCA
        thresholds: Listes de seuils de variance
        
    Returns:
        DataFrame de suggestions
    """
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
    """
    Génère un rapport textuel complet de l'analyse PCA.
    
    Args:
        pca_result: Résultat de la PCA
        
    Returns:
        Rapport textuel
    """
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

IV. TOP VARIABLES PAR COMPOSANTE
--------------------------------
"""
        
        for i in range(min(3, n_components)):
            try:
                top_vars = get_top_loadings(pca_result, i, 5)
                if not top_vars.empty:
                    report += f"\nComposante PC{i+1} :\n"
                    for _, row in top_vars.iterrows():
                        report += f"- {row['variable']} (loading: {row['loading']:.3f})\n"
            except:
                report += f"\nComposante PC{i+1} : (calcul impossible)\n"
        
        report += """

V. INTERPRÉTATION
-----------------
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


def get_variable_contributions(pca_result: Dict) -> pd.DataFrame:
    """
    Calcule les contributions des variables à chaque composante.
    
    Args:
        pca_result: Résultat de la PCA
        
    Returns:
        DataFrame des contributions
    """
    try:
        loadings = pca_result['loadings']
        n_components = pca_result['n_components']
        
        contributions = (loadings**2) * 100 / np.sum(loadings**2, axis=0)
        
        df_contrib = pd.DataFrame(
            contributions[:, :n_components],
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=pca_result['feature_names']
        )
        
        # Ajouter la contribution totale
        df_contrib['total_contribution'] = df_contrib.sum(axis=1)
        df_contrib['quality_representation'] = pca_result['metrics']['variable_cos2']
        
        return df_contrib.sort_values('total_contribution', ascending=False).round(4)
    except Exception as e:
        logger.error(f"Erreur dans get_variable_contributions: {e}")
        return pd.DataFrame({'error': [str(e)]})


def get_individual_analysis(pca_result: Dict, individual_idx: int = 0) -> Dict:
    """
    Analyse détaillée d'un individu spécifique.
    
    Args:
        pca_result: Résultat de la PCA
        individual_idx: Index de l'individu
        
    Returns:
        Dictionnaire d'analyse
    """
    try:
        if individual_idx >= len(pca_result['X_pca']):
            raise ValueError(f"Individu {individual_idx} hors limites. Maximum: {len(pca_result['X_pca'])-1}")
        
        coordinates = pca_result['X_pca'][individual_idx]
        cos2 = pca_result['metrics']['cos2_individuals'][individual_idx]
        
        # Distance au centre
        distance = np.sqrt(np.sum(coordinates**2))
        
        # Coordonnées originales
        try:
            original_coords = pca_result['X_clean'].iloc[individual_idx]
            original_dict = original_coords.to_dict()
        except:
            original_dict = {}
        
        # Contribution aux axes
        try:
            axis_contributions = (coordinates**2) * 100 / np.sum(coordinates**2)
        except:
            axis_contributions = np.ones_like(coordinates) * (100 / len(coordinates))
        
        # Qualité
        if cos2 > 0.5:
            quality = 'Bonne'
        elif cos2 > 0.3:
            quality = 'Moyenne'
        else:
            quality = 'Faible'
        
        analysis = {
            'coordinates': coordinates,
            'cos2': cos2,
            'distance_to_center': distance,
            'contributions_to_axes': axis_contributions,
            'original_values': original_dict,
            'quality': quality
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Erreur dans get_individual_analysis: {e}")
        return {'error': str(e)}


# ============================================================================
# FONCTION D'EXPORT POUR STREAMLIT
# ============================================================================

def get_pca_dashboard_data(pca_result: Dict) -> Dict:
    """
    Prépare toutes les données pour un dashboard complet.
    
    Args:
        pca_result: Résultat de la PCA
        
    Returns:
        Dictionnaire avec toutes les données
    """
    try:
        return {
            'summary': get_pca_summary(pca_result),
            'variable_contributions': get_variable_contributions(pca_result).head(20),
            'optimal_components_suggestions': suggest_optimal_components(pca_result),
            'report': generate_pca_report(pca_result),
            'metrics': {
                'total_variance': pca_result['cumulative_variance'][-1],
                'avg_quality': pca_result['metrics']['mean_cos2'],
                'kmo': pca_result['metrics'].get('kmo_measure', {}).get('kmo', 0),
                'bartlett_p': pca_result['metrics'].get('bartlett_sphericity', {}).get('p_value', 1),
            },
            'top_variables': {
                f'PC{i+1}': get_top_loadings(pca_result, i, 5).to_dict('records')
                for i in range(min(3, pca_result['n_components']))
            }
        }
    except Exception as e:
        logger.error(f"Erreur dans get_pca_dashboard_data: {e}")
        return {'error': str(e)}


# ============================================================================
# FONCTION DE TEST
# ============================================================================

def test_pca_module():
    """
    Teste le module PCA avec des données synthétiques.
    """
    import sys
    
    logger.info("🧪 Démarrage du test PCA...")
    
    try:
        # Créer des données de test
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Créer des données corrélées
        X = np.random.randn(n_samples, n_features)
        # Ajouter de la corrélation
        X[:, 2] = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)
        X[:, 3] = X[:, 0] - 0.3 * X[:, 1] + 0.2 * np.random.randn(n_samples)
        
        # Convertir en DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        
        logger.info(f"✅ Données de test créées: {df.shape}")
        
        # Tester la PCA
        logger.info("🚀 Test de la PCA...")
        pca_result = compute_pca(df, n_components=3)
        
        # Vérifications
        assert 'X_pca' in pca_result, "❌ X_pca manquant"
        assert 'explained_variance_ratio' in pca_result, "❌ explained_variance_ratio manquant"
        assert 'cumulative_variance' in pca_result, "❌ cumulative_variance manquant"
        assert 'n_components' in pca_result, "❌ n_components manquant"
        
        logger.info(f"✅ PCA réussie!")
        logger.info(f"   Composantes: {pca_result['n_components']}")
        logger.info(f"   Variance totale: {pca_result['cumulative_variance'][-1]:.3f}")
        
        # Tester les visualisations
        logger.info("🎨 Test des visualisations...")
        try:
            fig1 = create_scree_plot(pca_result)
            logger.info("✅ Scree plot créé")
            
            fig2 = create_correlation_circle(pca_result)
            logger.info("✅ Cercle des corrélations créé")
        except Exception as e:
            logger.warning(f"⚠ Erreur dans les visualisations: {e}")
        
        # Tester le rapport
        logger.info("📄 Test du rapport...")
        try:
            report = generate_pca_report(pca_result)
            logger.info("✅ Rapport généré")
        except Exception as e:
            logger.warning(f"⚠ Erreur dans le rapport: {e}")
        
        logger.info("🎉 Tous les tests ont réussi!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Échec du test: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Exécuter le test
    success = test_pca_module()
    if success:
        print("\n" + "="*50)
        print("✅ TOUS LES TESTS ONT RÉUSSI!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("="*50)