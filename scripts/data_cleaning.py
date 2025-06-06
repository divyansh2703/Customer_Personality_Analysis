import pandas as pd
import numpy as np
import os

# Load the dataset
df = pd.read_csv(r"C:\Users\doshi\Desktop\Data analyst projects\Customer_Personality_Analysis\data\raw\marketing_campaign.csv", sep="\t")

# Preview the data
print("Initial Shape:", df.shape)
print(df.head())

# Step 1: Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Step 2: Check for duplicates
duplicates = df.duplicated().sum()
print("Duplicates found:", duplicates)

# Step 3: Check for null values
nulls = df.isnull().sum()
print("\nMissing values:\n", nulls[nulls > 0])

# Step 4: Clean or fill missing values (e.g., income)
df = df.dropna(subset=["income"])  # Drop rows where income is missing

# Step 5: Convert date column
df["dt_customer"] = pd.to_datetime(df["dt_customer"], dayfirst=True)


# Step 6: Create new columns
df["age"] = 2024 - df["year_birth"]
df["total_spent"] = df[[
    "mntwines", "mntfruits", "mntmeatproducts", "mntfishproducts", "mntsweetproducts", "mntgoldprods"
]].sum(axis=1)

# Step 7: Fix categorical values
df["marital_status"] = df["marital_status"].replace({
    "Alone": "Single", "Absurd": "Single", "YOLO": "Single"
})
df["education"] = df["education"].replace({
    "2n Cycle": "Undergraduate"
})

# Step 8: Save cleaned data
os.makedirs("data/clean", exist_ok=True)
df.to_csv("data/clean/marketing_campaign_clean.csv", index=False)

print("\n✅ Cleaning completed. Cleaned dataset saved to 'data/clean/marketing_campaign_clean.csv'")

print(pd.to_datetime(df["dt_customer"], dayfirst=True).head())

