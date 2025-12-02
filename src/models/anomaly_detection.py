def generate_anomaly_report(
    df_raw: pd.DataFrame,
    anomaly_result: Dict,
    score_threshold: float = None
) -> str:
    """
    Génère un rapport détaillé en français des anomalies détectées.
    
    Returns:
        Rapport formaté en markdown
    """
    if score_threshold is None:
        score_threshold = np.percentile(anomaly_result['anomaly_scores'], 95)
    
    df_anomalies = analyze_anomalies(df_raw, anomaly_result, score_threshold)
    high_score_tx = df_anomalies[df_anomalies['is_above_threshold']]
    
    stats = get_anomaly_statistics(anomaly_result)
    
    report = f"""
# 📊 RAPPORT DE DÉTECTION D'ANOMALIES
*Généré le {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}*

## 📈 Résumé exécutif

### Données analysées
- **Transactions totales** : {stats['n_total']:,}
- **Features utilisées** : {anomaly_result['X_scaled'].shape[1]}
- **Période analysée** : {df_raw['transaction_date'].min().date() if 'transaction_date' in df_raw.columns else 'Non spécifiée'} au {df_raw['transaction_date'].max().date() if 'transaction_date' in df_raw.columns else 'Non spécifiée'}

### Résultats de détection
- **Anomalies détectées** : {stats['n_anomalies']:,} ({stats['pct_anomalies']:.1f}%)
- **Score d'anomalie moyen** : {stats['score_mean']:.3f}
- **Score maximum** : {stats['score_max']:.3f}
- **Seuil de détection (Q95)** : {stats['score_q95']:.3f}
- **Transactions au-dessus du seuil** : {len(high_score_tx):,}

## 🔍 Analyse des anomalies

### Top 5 transactions les plus suspectes
"""
    
    if not high_score_tx.empty:
        top5 = high_score_tx.head(5)
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            report += f"\n{i}. **Transaction {row.get('transaction_id', idx)}** : "
            report += f"Score = {row.get('anomaly_score', 0):.3f}, "
            if 'product_amount' in row:
                report += f"Montant = {row['product_amount']:.2f}€"
            if 'product_category' in row:
                report += f", Catégorie = {row['product_category']}"
            report += f", User = {row.get('user_id', 'N/A')}"
    
    report += """

## 📊 Distribution statistique

### Quartiles des scores
- **Q25 (25%)** : {:.3f}
- **Q50 (Médiane)** : {:.3f}
- **Q75 (75%)** : {:.3f}
- **Q95 (95%)** : {:.3f}
- **Q99 (99%)** : {:.3f}

## 🎯 Recommandations

1. **Vérification manuelle** des transactions avec score > {:.3f}
2. **Analyse approfondie** des utilisateurs récurrents dans les anomalies
3. **Révision des règles métier** pour les catégories à risque
4. **Surveillance continue** avec mise à jour hebdomadaire du modèle

## 📈 Métriques de qualité

- **Séparation scores** : {:.2%} (écart moyen normal/anomalie)
- **Stabilité détection** : Contamination utilisée = {:.1%}
- **Capacité prédictive** : Modèle entraîné sur {:,} échantillons

---
*Ce rapport a été généré automatiquement par le système de détection d'anomalies Fintech*
""".format(
        stats['score_q25'],
        stats['score_median'],
        stats['score_q75'],
        stats['score_q95'],
        stats['score_q99'],
        score_threshold,
        (stats['score_mean'] - np.mean(anomaly_result['anomaly_scores'][~anomaly_result['is_anomaly']])) / stats['score_mean'] if 'score_mean' in stats else 0,
        anomaly_result.get('contamination', 0.02),
        len(anomaly_result['anomaly_scores'])
    )
    
    return report