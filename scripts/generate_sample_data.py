"""
generate_sample_data.py
------------------------
Generates a synthetic customer/marketing dataset that mirrors the schema
used in this project (based on the classic "Customer Personality Analysis"
structure: demographics + purchase history + campaign response).

This is provided so the pipeline runs end-to-end out of the box.
To reproduce the exact published results, replace
`data/raw/marketing_campaign.csv` with your own dataset using the same
column names.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 2240  # matches original dataset scale

education_levels = ["Graduation", "PhD", "Master", "Basic", "2n Cycle"]
marital_status = ["Married", "Together", "Single", "Divorced", "Widow"]

# ---- Base demographics ----
df = pd.DataFrame({
    "ID": np.arange(1, N + 1),
    "Year_Birth": np.random.randint(1940, 2000, N),
    "Education": np.random.choice(education_levels, N, p=[0.5, 0.15, 0.2, 0.1, 0.05]),
    "Marital_Status": np.random.choice(marital_status, N, p=[0.38, 0.26, 0.22, 0.1, 0.04]),
    "Income": np.round(np.random.lognormal(mean=10.7, sigma=0.4, size=N), 2),
    "Kidhome": np.random.choice([0, 1, 2], N, p=[0.55, 0.35, 0.1]),
    "Teenhome": np.random.choice([0, 1, 2], N, p=[0.55, 0.35, 0.1]),
})

# Inject missing income values (to be dropped in cleaning -> mirrors 24 dropped rows)
missing_idx = np.random.choice(df.index, 24, replace=False)
df.loc[missing_idx, "Income"] = np.nan

# ---- Enrollment date & recency ----
start_date = datetime(2012, 1, 1)
end_date = datetime(2014, 12, 31)
df["Dt_Customer"] = [
    (start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days)))
    .strftime("%d-%m-%Y")
    for _ in range(N)
]
df["Recency"] = np.random.randint(0, 100, N)

# ---- Spending categories (skewed, income-correlated) ----
income_scale = (df["Income"].fillna(df["Income"].median()) / df["Income"].median())

def spend_col(base, noise=0.6):
    return np.round(np.clip(base * income_scale * np.random.lognormal(0, noise, N), 0, None), 2)

df["MntFruits"] = spend_col(20)
df["MntFishProducts"] = spend_col(28)
df["MntSweetProducts"] = spend_col(20)
df["MntGoldProds"] = spend_col(35)

# Wine & meat spending share a "premium buyer" latent factor,
# producing a strong cross-category affinity (~0.68), matching the
# bundle-recommendation pattern this pipeline is designed to surface.
df["MntWines"] = spend_col(180, noise=0.32)
df["MntMeatProducts"] = spend_col(120, noise=0.32)
premium_affinity = np.random.lognormal(mean=0, sigma=0.55, size=N)
df["MntWines"] = np.round(df["MntWines"] * (0.3 + 1.3 * premium_affinity), 2)
df["MntMeatProducts"] = np.round(df["MntMeatProducts"] * (0.3 + 1.3 * premium_affinity), 2)

# ---- Purchase channels ----
df["NumDealsPurchases"] = np.random.poisson(2, N)
df["NumWebPurchases"] = np.random.poisson(4, N)
df["NumCatalogPurchases"] = np.random.poisson(2, N)
df["NumStorePurchases"] = np.random.poisson(5, N)
df["NumWebVisitsMonth"] = np.random.poisson(5, N)

# ---- Campaign history ----
for c in ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]:
    df[c] = np.random.choice([0, 1], N, p=[0.9, 0.1])

df["Complain"] = np.random.choice([0, 1], N, p=[0.99, 0.01])
df["Z_CostContact"] = 3
df["Z_Revenue"] = 11

# ---- Target: Response to latest campaign ----
# Built from a logistic combination of behavioral signals so the
# downstream classifier has genuine, learnable structure to find.
total_spend_tmp = df[["MntWines", "MntFruits", "MntMeatProducts",
                       "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)
past_campaigns_tmp = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                          "AcceptedCmp4", "AcceptedCmp5"]].sum(axis=1)


def z(s):
    return (s - s.mean()) / s.std()


logit = (
    -1.9
    + 1.4 * z(total_spend_tmp)
    + 0.7 * z(df["Income"].fillna(df["Income"].median()))
    - 0.6 * z(df["Recency"])
    + 0.9 * z(past_campaigns_tmp)
    + 0.4 * z(df["NumWebVisitsMonth"])
    - 0.3 * z(df["Kidhome"] + df["Teenhome"])
)
prob = 1 / (1 + np.exp(-logit))
df["Response"] = np.random.binomial(1, prob)

# Shuffle column order to mirror raw export style
cols = ["ID", "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
        "Dt_Customer", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases",
        "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
        "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "AcceptedCmp1", "AcceptedCmp2",
        "Complain", "Z_CostContact", "Z_Revenue", "Response"]
df = df[cols]

df.to_csv("data/raw/marketing_campaign.csv", index=False, sep="\t")
print(f"Generated {len(df)} rows, {len(df.columns)} columns -> data/raw/marketing_campaign.csv")
