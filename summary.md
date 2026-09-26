# Pipeline Run — Results Summary

_Generated automatically by `src/run_pipeline.py` on the dataset in `data/raw/marketing_campaign.csv`._

## Customer Segmentation
- High-value segment: **382 customers** (17.3% of the base)
- Revenue share of that segment: **34.7%** of total spend ($1,631,901)
- Silhouette score (K=4): 0.122

## Campaign Response Model
- ROC-AUC: **0.876**
- Baseline conversion rate: 22.1%
- Targeted conversion rate (top 551 customers by predicted probability): **77.9%** (3.52x lift)

## Basket / Cross-Sell Analysis
- Top affinity pair: **MntWines & MntMeatProducts** (correlation 0.76)
- Bundle target segment size: 241 customers
- Projected AOV for that segment: $67.98 → **$85.28** (25.5% uplift)

## Figures
- `outputs/figures/elbow_silhouette.png`
- `outputs/figures/pca_clusters.png`
- `outputs/figures/roc_curve.png`
- `outputs/figures/shap_summary.png`
- `outputs/figures/correlation_heatmap.png`

> Note: this run uses the synthetic sample dataset included in `data/raw/`. Replace it with your own dataset (same column schema) to reproduce results on real data.