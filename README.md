# Customer Marketing Intelligence Engine

A complete end-to-end customer analytics pipeline built in Python. This project transforms raw demographic and campaign data into structured customer segments, predictive response models, and cross-sell strategies.


---

## 📊 Key Results & Analytical Findings

| Metric / Task | Baseline / Raw Input | Model / Analytical Result |
|---|---|---|
| Dataset Scale | 2,240 records (29 columns) | 2,216 clean records (31 columns) |
| High-Value Customer Segment | Undefined | 399 users (18%) driving $420k (52%) |
| Campaign Conversion Rate | 14.9% baseline overall | 26.3% targeted (+11.4% / 1.76x lift) |
| Response Model ROC-AUC | Random (0.50) | XGBoost Classifier (0.84 ROC-AUC) |
| Projected Average Order Value | $112.00 per order | $136.64 per order (+$24.64 / +22%) |

---

## 🔍 Highlights & Impact

- **Automated Data Processing & Feature Engineering** — Built a modular Python ETL script using `pandas` to clean 2,216 customer records (removing 24 missing-income rows); engineered total spending across 6 categories and derived customer tenure/age variables.

- **Behavioral Customer Segmentation** — Evaluated K-Means clustering (K=4) with Principal Component Analysis (PCA) and Silhouette Scores to isolate a core "High-Value" cluster of 399 customers (18% of sample) generating $420,000 of total $808,000 spend (52%).

- **Predictive Campaign Modeling** — Trained an XGBoost response classifier to identify high-probability respondents, achieving an AUC-ROC of 0.84 and raising conversion rates to 26.3% (vs. 14.9% baseline across all 2,216 records).

- **Basket Analysis & Cross-Selling** — Analyzed correlation matrices across spending categories, uncovering a 0.68 affinity score between premium wine and meat purchases to support bundle recommendations projected to increase Average Order Value from $112.00 to $136.64 (+$24.64 per transaction).

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn (K-Means, PCA), XGBoost, LightGBM, SHAP
- **Visualization:** Matplotlib / Seaborn
- **Environment:** Jupyter Notebook

---

## 📁 Repository Structure

```
customer-marketing-intelligence-engine/
├── data/
│   ├── raw/                  # Original raw dataset
│   └── processed/            # Cleaned dataset (2,216 records)
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_customer_segmentation.ipynb
│   ├── 04_response_modeling.ipynb
│   └── 05_basket_analysis.ipynb
├── src/
│   ├── etl.py                 # Data cleaning & feature engineering pipeline
│   ├── segmentation.py        # K-Means + PCA clustering
│   ├── response_model.py      # XGBoost classifier training/evaluation
│   └── basket_analysis.py     # Cross-sell correlation analysis
├── outputs/
│   ├── figures/                # Charts and visualizations
│   └── models/                 # Saved model artifacts
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/customer-marketing-intelligence-engine.git
cd customer-marketing-intelligence-engine
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Run the full pipeline
python src/etl.py
python src/segmentation.py
python src/response_model.py
python src/basket_analysis.py
```

Or explore the analysis step-by-step in the `notebooks/` directory.

---

## 📈 Methodology

1. **Data Cleaning** — Removed missing-income records, handled outliers, and standardized data types across the 2,240-record raw dataset.
2. **Feature Engineering** — Aggregated spending across 6 product categories and derived tenure/age variables from enrollment and birth dates.
3. **Segmentation** — Applied K-Means (K=4) on PCA-reduced features, validated with Silhouette Scores, to identify a distinct high-value customer segment.
4. **Predictive Modeling** — Trained and tuned an XGBoost classifier to predict campaign response probability, benchmarked against baseline conversion rates.
5. **Cross-Sell Analysis** — Built a correlation matrix across spending categories to surface product affinities for bundling strategies.

---

## 💡 Business Impact

- Enables targeted marketing by identifying the **18% of customers driving 52% of revenue**.
- Improves campaign efficiency with a **1.76x lift in conversion rate** over blanket targeting.
- Supports a **data-backed bundling strategy** (wine + meat) with a projected **22% increase in Average Order Value**.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 📬 Contact

Feel free to reach out with questions or feedback about this project.
