"""
response_model.py
------------------
Trains an XGBoost classifier to predict campaign response probability
and evaluates it against a baseline conversion rate. Saves the ROC
curve, a SHAP feature-importance summary, and the trained model.

Input:  data/processed/customers_with_segments.csv
Output: outputs/figures/roc_curve.png
        outputs/figures/shap_summary.png
        outputs/models/xgb_response_model.pkl
        outputs/results/response_model_metrics.json
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

DATA_PATH = "data/processed/customers_with_segments.csv"
RANDOM_STATE = 42

FEATURES = [
    "Age", "Income", "Total_Spending", "Total_Children", "Recency",
    "Customer_Tenure_Days", "Total_Purchases", "Total_Campaigns_Accepted",
    "NumWebVisitsMonth", "Cluster",
]
TARGET = "Response"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    return df, X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    return model, X_train, X_test, y_test, y_proba, auc


def plot_roc(y_test, y_proba, auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.2f})", color="steelblue")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Campaign Response Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/roc_curve.png", dpi=150)
    plt.close()


def plot_shap_summary(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def targeted_conversion_lift(df, model, top_pct=0.25):
    """Simulate targeting the top-N% highest-probability customers."""
    X_all = df[FEATURES]
    df = df.copy()
    df["response_proba"] = model.predict_proba(X_all)[:, 1]

    baseline_rate = df[TARGET].mean()

    n_top = int(len(df) * top_pct)
    top_customers = df.sort_values("response_proba", ascending=False).head(n_top)
    targeted_rate = top_customers[TARGET].mean()

    return baseline_rate, targeted_rate, n_top


def run():
    df, X, y = load_data()
    model, X_train, X_test, y_test, y_proba, auc = train_model(X, y)

    plot_roc(y_test, y_proba, auc)
    plot_shap_summary(model, X_train)

    baseline_rate, targeted_rate, n_top = targeted_conversion_lift(df, model)

    with open("outputs/models/xgb_response_model.pkl", "wb") as f:
        pickle.dump(model, f)

    result = {
        "roc_auc": round(float(auc), 3),
        "baseline_conversion_rate_pct": round(float(baseline_rate) * 100, 1),
        "targeted_conversion_rate_pct": round(float(targeted_rate) * 100, 1),
        "lift_x": round(float(targeted_rate / baseline_rate), 2) if baseline_rate > 0 else None,
        "customers_targeted": n_top,
    }

    with open("outputs/results/response_model_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return model, result


if __name__ == "__main__":
    run()
