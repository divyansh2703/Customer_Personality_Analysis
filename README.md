# Customer Marketing Data Cleaning and Spending Feature Engineering

A customer marketing data preparation project focused on making demographic, spending and campaign fields easier to analyse consistently. The current public implementation is a pandas cleaning script with source and cleaned CSV files. Segmentation, campaign attribution and return on advertising spend analysis have not been implemented in the committed code.

## The question

How can customer marketing records be cleaned and structured before analysing spending patterns or campaign response?

## Tools and methods

Python, pandas, Data cleaning, Feature engineering, Customer analytics.

## Work in this repository

1. Standardised column names and inspected duplicate records and missing values.
2. Removed records with missing income and parsed customer enrolment dates.
3. Normalised selected marital status and education labels.
4. Created an age field using a fixed 2024 reference year and a total spending feature across six product spending columns.
5. Saved a cleaned customer table for later exploration.

## Evidence and scope

| Measure | Recorded value |
| --- | --- |
| Raw customer records | 2,240 |
| Raw columns | 29 |
| Records with missing income | 24 |
| Cleaned records | 2,216 |
| Cleaned columns | 31 |
| Spending categories summed | 6 |

## Repository guide

| File or folder | Purpose |
| --- | --- |
| [scripts/data_cleaning.py](https://github.com/divyansh2703/Customer_Personality_Analysis/blob/main/scripts/data_cleaning.py) | Transformation logic |
| [data/raw/marketing_campaign.csv](https://github.com/divyansh2703/Customer_Personality_Analysis/blob/main/data/raw/marketing_campaign.csv) | Raw tab separated source |
| [scripts/data/clean/marketing_campaign_clean.csv](https://github.com/divyansh2703/Customer_Personality_Analysis/blob/main/scripts/data/clean/marketing_campaign_clean.csv) | Committed cleaned output |
| [requirements.txt](https://github.com/divyansh2703/Customer_Personality_Analysis/blob/main/requirements.txt) | Listed Python dependencies |

## Getting started

Create a dedicated Python environment and install the listed dependencies with `python -m pip install -r requirements.txt`. Update the absolute Windows source path in `scripts/data_cleaning.py` to your local copy of `data/raw/marketing_campaign.csv`.

Run the script only after reviewing that path:

```bash
python scripts/data_cleaning.py
```

The script writes to `data/clean/` relative to the current working directory. The historical committed output is under `scripts/data/clean/`, so the output location depends on where the script is launched. Choose and document the working directory before comparing outputs.

## Current limitations

1. The age field is calculated as 2024 minus birth year. It is not current age in 2026.
2. The script counts duplicates but does not remove them.
3. Dropping missing income records is an implemented choice whose potential selection effect should be assessed before modelling.
4. The source table does not establish a completed comparison of social media, influencer or search advertising effectiveness. No segmentation or marketing ROI result is claimed.

## Next steps

1. Replace the absolute source path with configuration and document the age reference date.
2. Investigate spending and response patterns before selecting a segmentation method.
3. Document data provenance, removal decisions and analytical conclusions.

## Authors and reuse

Divyansh Doshi.

Documentation reviewed against the public repository on 7 September 2026. Counts are taken from the named saved artifacts or directly inspected CSVs; this review did not rerun model training or validate a complete deployment. No source code licence was found in the reviewed project tree. Data and third party material may have separate terms.
