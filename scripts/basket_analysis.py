"""
basket_analysis.py
-------------------
Analyzes correlations across product spending categories to surface
cross-sell / bundling opportunities, and projects the Average Order
Value (AOV) uplift from targeting the top-affinity product pair.

Input:  data/processed/customers_with_segments.csv
Output: outputs/figures/correlation_heatmap.png
        outputs/results/basket_analysis_metrics.json
"""

import json
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/customers_with_segments.csv"
SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]


def compute_correlations(df: pd.DataFrame):
    corr = df[SPEND_COLS].corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spending Category Correlation Matrix")
    plt.tight_layout()
    plt.savefig("outputs/figures/correlation_heatmap.png", dpi=150)
    plt.close()

    return corr


def top_affinity_pair(corr: pd.DataFrame):
    pairs = []
    for a, b in itertools.combinations(SPEND_COLS, 2):
        pairs.append((a, b, corr.loc[a, b]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[0]


def project_aov_uplift(df: pd.DataFrame, pair, avg_bundle_discount=0.15):
    """
    Estimate the AOV uplift a bundle recommendation would deliver for
    its actual target segment: customers who over-index on ONE product
    in the top-affinity pair but are below-median on the other. These
    are the natural cross-sell targets. We project their AOV if they
    were bundled up to the average spend level of existing high
    spenders on the missing product, net of a bundle discount.
    """
    col_a, col_b, _ = pair

    df = df.copy()
    df["n_orders_proxy"] = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    df["n_orders_proxy"] = df["n_orders_proxy"].replace(0, 1)
    df["total_spend"] = df[SPEND_COLS].sum(axis=1)
    df["customer_aov"] = df["total_spend"] / df["n_orders_proxy"]

    med_a, med_b = df[col_a].median(), df[col_b].median()
    candidates = df[(df[col_a] > med_a) & (df[col_b] <= med_b)].copy()

    baseline_aov = candidates["customer_aov"].mean()

    avg_high_spend_b = df[df[col_b] > med_b][col_b].mean()
    bundle_add = avg_high_spend_b * (1 - avg_bundle_discount) - candidates[col_b]

    projected_aov = ((candidates["total_spend"] + bundle_add) / candidates["n_orders_proxy"]).mean()
    convert_n = len(candidates)

    return baseline_aov, projected_aov, convert_n


def run():
    df = pd.read_csv(DATA_PATH)
    corr = compute_correlations(df)
    pair = top_affinity_pair(corr)
    baseline_aov, projected_aov, convert_n = project_aov_uplift(df, pair)

    result = {
        "top_affinity_pair": [pair[0], pair[1]],
        "affinity_score": round(float(pair[2]), 2),
        "baseline_aov": round(float(baseline_aov), 2),
        "projected_aov": round(float(projected_aov), 2),
        "aov_uplift": round(float(projected_aov - baseline_aov), 2),
        "aov_uplift_pct": round(float((projected_aov / baseline_aov - 1) * 100), 1),
        "bundle_target_segment_size": convert_n,
    }

    with open("outputs/results/basket_analysis_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
