"""
run_pipeline.py
----------------
Runs the full pipeline end-to-end:
  1. ETL (clean + engineer features)
  2. Customer segmentation (K-Means + PCA)
  3. Campaign response modeling (XGBoost)
  4. Basket / cross-sell analysis

Writes a consolidated results file to outputs/results/summary.md.
"""

import json
import etl
import segmentation
import response_model
import basket_analysis


def main():
    print("=" * 60)
    print("STEP 1/4 — ETL")
    print("=" * 60)
    etl.run()

    print("\n" + "=" * 60)
    print("STEP 2/4 — Customer Segmentation")
    print("=" * 60)
    _, seg_summary, seg_metrics = segmentation.run()

    print("\n" + "=" * 60)
    print("STEP 3/4 — Campaign Response Modeling")
    print("=" * 60)
    _, model_metrics = response_model.run()

    print("\n" + "=" * 60)
    print("STEP 4/4 — Basket / Cross-Sell Analysis")
    print("=" * 60)
    basket_metrics = basket_analysis.run()

    write_summary(seg_metrics, model_metrics, basket_metrics)
    print("\nAll steps complete. See outputs/results/summary.md for a consolidated report.")


def write_summary(seg, model, basket):
    lines = [
        "# Pipeline Run — Results Summary",
        "",
        "_Generated automatically by `src/run_pipeline.py` on the dataset in "
        "`data/raw/marketing_campaign.csv`._",
        "",
        "## Customer Segmentation",
        f"- High-value segment: **{seg['high_value_customers']} customers** "
        f"({seg['high_value_pct_of_customers']}% of the base)",
        f"- Revenue share of that segment: **{seg['high_value_pct_of_revenue']}%** "
        f"of total spend (${seg['total_revenue']:,.0f})",
        f"- Silhouette score (K=4): {seg['silhouette_score']}",
        "",
        "## Campaign Response Model",
        f"- ROC-AUC: **{model['roc_auc']}**",
        f"- Baseline conversion rate: {model['baseline_conversion_rate_pct']}%",
        f"- Targeted conversion rate (top {model['customers_targeted']} customers by "
        f"predicted probability): **{model['targeted_conversion_rate_pct']}%** "
        f"({model['lift_x']}x lift)",
        "",
        "## Basket / Cross-Sell Analysis",
        f"- Top affinity pair: **{basket['top_affinity_pair'][0]} & "
        f"{basket['top_affinity_pair'][1]}** (correlation {basket['affinity_score']})",
        f"- Bundle target segment size: {basket['bundle_target_segment_size']} customers",
        f"- Projected AOV for that segment: ${basket['baseline_aov']} → "
        f"**${basket['projected_aov']}** ({basket['aov_uplift_pct']}% uplift)",
        "",
        "## Figures",
        "- `outputs/figures/elbow_silhouette.png`",
        "- `outputs/figures/pca_clusters.png`",
        "- `outputs/figures/roc_curve.png`",
        "- `outputs/figures/shap_summary.png`",
        "- `outputs/figures/correlation_heatmap.png`",
        "",
        "> Note: this run uses the synthetic sample dataset included in "
        "`data/raw/`. Replace it with your own dataset (same column schema) "
        "to reproduce results on real data.",
    ]

    with open("outputs/results/summary.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
