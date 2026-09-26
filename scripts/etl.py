"""
etl.py
------
Loads the raw marketing campaign dataset, cleans it, and engineers
features used downstream by segmentation, response modeling, and
basket analysis.

Input:  data/raw/marketing_campaign.csv
Output: data/processed/cleaned_customers.csv
"""

import pandas as pd
from datetime import datetime

RAW_PATH = "data/raw/marketing_campaign.csv"
PROCESSED_PATH = "data/processed/cleaned_customers.csv"

SPEND_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
              "MntFishProducts", "MntSweetProducts", "MntGoldProds"]


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # Drop rows with missing income
    df = df.dropna(subset=["Income"]).copy()

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Remove extreme outliers (age > 100, income > 99.5th percentile)
    df["Age"] = datetime.now().year - df["Year_Birth"]
    df = df[df["Age"] <= 100]
    income_cap = df["Income"].quantile(0.995)
    df = df[df["Income"] <= income_cap]

    dropped = before - len(df)
    print(f"Cleaning: removed {dropped} rows ({before} -> {len(df)})")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total spend across all categories
    df["Total_Spending"] = df[SPEND_COLS].sum(axis=1)

    # Household size & children
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # Customer tenure in days since enrollment
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    reference_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

    # Total purchases across all channels
    df["Total_Purchases"] = (
        df["NumDealsPurchases"] + df["NumWebPurchases"]
        + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    )

    # Total campaigns accepted historically (prior to the target campaign)
    cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    df["Total_Campaigns_Accepted"] = df[cmp_cols].sum(axis=1)

    return df


def run():
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned dataset -> {PROCESSED_PATH}")
    print(f"Final shape: {df.shape[0]} records, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    run()
