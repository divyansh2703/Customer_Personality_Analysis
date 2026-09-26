Customer Marketing Intelligence EngineA complete end-to-end customer analytics pipeline built in Python. This project transforms raw demographic and campaign data into structured customer segments, predictive response models, and cross-sell strategies.

Key Results & Analytical

Findings+-----------------------------------------------------------------------------------------------+
|  Metric / Task             | Baseline / Raw Input       | Model / Analytical Result           |
+----------------------------+----------------------------+-------------------------------------+
| Dataset Scale              | 2,240 records (29 columns) | 2,216 clean records (31 columns)    |
| High-Value Customer Segment| Undefined                  | 399 users (18%) driving $420k (52%) |
| Campaign Conversion Rate   | 14.9% baseline overall     | 26.3% targeted (+11.4% / 1.76x lift)|
| Response Model ROC-AUC     | Random (0.50)              | XGBoost Classifier (0.84 ROC-AUC)   |
| Projected Average Order    | $112.00 per order          | $136.64 per order (+$24.64 / +22%)  |
+----------------------------+----------------------------+-------------------------------------+


Highlights & ImpactAutomated Data Processing & Feature Engineering: 
Built a modular Python ETL script using pandas to clean 2,216 customer records (removing 24 missing-income rows); engineered total spending across 6 categories and derived customer tenure/age variables.
Behavioral Customer Segmentation: Evaluated K-Means clustering ($K=4$) with Principal Component Analysis (PCA) and Silhouette Scores to isolate a core "High-Value" cluster of 399 customers (18% of sample) generating $420,000 of total $808,000 spend (52%).
Predictive Campaign Modeling: Trained an XGBoost response classifier to identify high-probability respondents, achieving an AUC-ROC of 0.84 and raising conversion rates to 26.3% (vs. 14.9% baseline across all 2,216 records).
Basket Analysis & Cross-Selling: Analyzed correlation matrices across spending categories, uncovering a 0.68 affinity score between premium wine and meat purchases to support bundle recommendations projected to increase Average Order Value from $112.00 to $136.64 (+$24.64 per transactions
