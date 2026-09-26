"""
segmentation.py
----------------
Behavioral customer segmentation using K-Means clustering on
PCA-reduced features. Identifies the "High-Value" customer cluster
and saves diagnostic plots + a segment summary.

Input:  data/processed/cleaned_customers.csv
Output: outputs/figures/elbow_silhouette.png
        outputs/figures/pca_clusters.png
        outputs/results/segmentation_summary.csv
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROCESSED_PATH = "data/processed/cleaned_customers.csv"
K = 4
RANDOM_STATE = 42

FEATURES = [
    "Age", "Income", "Total_Spending", "Total_Children",
    "Customer_Tenure_Days", "Total_Purchases", "Recency",
]


def prepare_features(df: pd.DataFrame):
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def evaluate_k_range(X_scaled, k_range=range(2, 8)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(list(k_range), inertias, "o-", color="steelblue", label="Inertia")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia", color="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(list(k_range), silhouettes, "s--", color="darkorange", label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", color="darkorange")

    plt.title("Elbow Method & Silhouette Score by K")
    fig.tight_layout()
    plt.savefig("outputs/figures/elbow_silhouette.png", dpi=150)
    plt.close()

    return dict(zip(k_range, silhouettes))


def run_clustering(df: pd.DataFrame, X_scaled):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_scaled)

    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, df["Cluster"])

    plot_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    plot_df["Cluster"] = df["Cluster"].values

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster", palette="viridis", s=40)
    plt.title(f"Customer Segments (K={K}) — PCA Projection")
    plt.tight_layout()
    plt.savefig("outputs/figures/pca_clusters.png", dpi=150)
    plt.close()

    return df, sil_score


def summarize_segments(df: pd.DataFrame):
    summary = df.groupby("Cluster").agg(
        n_customers=("Cluster", "size"),
        avg_income=("Income", "mean"),
        avg_spending=("Total_Spending", "mean"),
        total_spending=("Total_Spending", "sum"),
        avg_age=("Age", "mean"),
    ).round(1)

    summary["pct_of_customers"] = (summary["n_customers"] / summary["n_customers"].sum() * 100).round(1)
    summary["pct_of_revenue"] = (summary["total_spending"] / summary["total_spending"].sum() * 100).round(1)

    high_value_cluster = summary["avg_spending"].idxmax()
    summary["segment_label"] = "Standard"
    summary.loc[high_value_cluster, "segment_label"] = "High-Value"

    summary.to_csv("outputs/results/segmentation_summary.csv")
    return summary, high_value_cluster


def run():
    df = pd.read_csv(PROCESSED_PATH)
    X_scaled = prepare_features(df)

    evaluate_k_range(X_scaled)
    df, sil_score = run_clustering(df, X_scaled)
    summary, hv_cluster = summarize_segments(df)

    hv_row = summary.loc[hv_cluster]
    result = {
        "silhouette_score": round(sil_score, 3),
        "high_value_cluster": int(hv_cluster),
        "high_value_customers": int(hv_row["n_customers"]),
        "high_value_pct_of_customers": float(hv_row["pct_of_customers"]),
        "high_value_pct_of_revenue": float(hv_row["pct_of_revenue"]),
        "total_revenue": float(df["Total_Spending"].sum()),
    }

    with open("outputs/results/segmentation_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Segmentation summary:\n", summary)
    print("\nKey metrics:\n", json.dumps(result, indent=2))

    df.to_csv("data/processed/customers_with_segments.csv", index=False)
    return df, summary, result


if __name__ == "__main__":
    run()
