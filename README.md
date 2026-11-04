# Predicting Customer Churn in a Telecom Company

Exploratory data analysis (EDA) and baseline machine-learning models for a telecom customer dataset, to understand which account, billing, usage and service characteristics are associated with customers leaving (churning) and to predict who is likely to churn.


## Project Overview

Customer churn is costly for telecom providers: retaining an existing customer is generally cheaper than acquiring a new one. This project explores a customer-level dataset to answer:

- What share of customers churn?
- Which variables differ most between churned and retained customers?
- How are usage, billing, complaints and delinquency related to each other?

- How well can simple models predict churn, and which features matter most?

## Dataset

| Property | Value |
|---|---|
| File | `Telco_Churn_Data.csv` |
| Rows | 4,708 customers |
| Columns | 15 |
| Target | `Target Churn` (label) / `Target Code` (0 = No Churn, 1 = Churn) |
| Class balance | ~48.6% Churn, ~51.4% No Churn |

### Data dictionary

| Column | Type | Description |
|---|---|---|
| `Target Churn` | categorical | Churn status as text (`Churn` / `No Churn`) |
| `Target Code` | binary | Numeric encoding of the target (1 = Churn) |
| `Current Bill Amt` | numeric | Current bill amount |
| `Avg Calls` | numeric | Average number of calls |
| `Avg Calls Weekdays` | numeric | Average number of weekday calls |
| `Account Age` | numeric | Age of the account (months) |
| `Percent Increase MOM` | numeric | Month-over-month percentage change |
| `Acct Plan Subtype` | categorical | Plan tier (`Silver`, `Gold`) |
| `Complaint Code` | categorical | Reason code of the customer's complaint |
| `Avg Days Delinquent` | numeric | Average number of days payments are late |
| `Current TechSupComplaints` | count | Current technical-support complaints |
| `Current Days OpenWorkOrders` | numeric | Days that work orders have been open |
| `Equipment Age` | numeric | Age of customer equipment |
| `Condition of Current Handset` | categorical | Handset condition code (1–3) |
| `Avg Hours WorkOrderOpenned` | numeric | Average hours a work order stays open |

> Units for some columns (e.g. currency, age) are not documented in the file — please confirm with the data provider and update this table.

## Repository Structure

```
.
├── Customer_Churn.ipynb      # EDA + modelling notebook
├── customer_churn.py         # Same code exported as a plain script
├── Telco_Churn_Data.csv      # Dataset
├── requirements.txt          # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

## Workflow

1. **Load data** – import libraries and read the CSV.
2. **Inspect** – shape, data types, summary statistics.
3. **Clean** – rename columns (with a check that every name exists); fix data types.
4. **Handle missing values** – mode imputation for exploration; inside the pipeline for modelling (fit on training data only).
5. **Data quality checks** – duplicates, negative bills, outliers, field consistency.
6. **Explore the target** – class balance and feature means by churn status.
7. **Correlation analysis** – correlation matrix and heatmap for numeric variables.
8. **Univariate & bivariate analysis** – histograms, churn rate by category (with group sizes), and density plots by churn status.
9. **Modelling** – stratified 80/20 split; Logistic Regression, Random Forest and Gradient Boosting in scikit-learn pipelines; evaluation with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves and feature importance.

## Key Findings

- **Delinquency is the strongest signal.** `Avg Days Delinquent` has the highest correlation with churn (r ≈ 0.46). Churned customers average about 19 days delinquent versus about 9 days for retained customers.
- **Usage variables overlap heavily.** `Avg Calls` and `Avg Calls Weekdays` are strongly correlated (r ≈ 0.73), and both are moderately related to `Current Bill Amt` (r ≈ 0.35 and 0.43). Expect multicollinearity if both are used in a linear model.
- **Most other numeric features have weak linear correlation with churn** (|r| < 0.10), including account age (r ≈ −0.09) and MoM change (r ≈ −0.06). Non-linear or categorical effects (plan type, complaint code) should be tested in modelling.
- **The classes are roughly balanced**, so accuracy is a reasonable starting metric, though precision/recall/ROC-AUC are still recommended.

## Model Results

Test-set results (942 customers, random seed 42; figures may vary slightly across library versions):

| Model | CV ROC-AUC (train) | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.845 | 0.786 | 0.794 | 0.755 | 0.774 | 0.857 |
| Random Forest | 0.883 | 0.815 | 0.795 | 0.836 | 0.815 | 0.895 |
| Gradient Boosting | 0.885 | 0.803 | 0.786 | 0.817 | 0.801 | 0.888 |

Tree-based models outperform logistic regression, which suggests non-linear relationships. In the Random Forest, `Avg_Days_Delinquent` and `Percent_Increase_MOM` are the most important features. `Percent_Increase_MOM` has almost no linear correlation with churn, so its importance is worth investigating further.

## Data Quality Notes

- **Missing values:** `Condition of Current Handset` (444 rows, ~9.4%) and `Complaint Code` (7 rows) contain nulls.
- **Outliers:** several fields are heavily right-skewed with extreme maxima (e.g. `Current Bill Amt` up to ~325k, `Percent Increase MOM` up to ~210), and `Current Bill Amt` contains a few negative values (min −690). These should be reviewed before modelling.
- **Unusual usage values:** `Avg Calls Weekdays` exceeds `Avg Calls` in ~96% of rows, which is counter-intuitive for a subset of the same calls. Confirm the definitions of these two fields with the data owner.
- **Redundant target columns:** `Target Churn` and `Target Code` encode the same information — use only one as the label and **drop the other from the feature set** to avoid target leakage.
- No duplicate rows were found.

## Tech Stack

- **Python** – pandas, NumPy
- **Visualisation** – Matplotlib, Seaborn
- **Modelling** – scikit-learn
- **Environment** – Jupyter Notebook


## Author

**Yaw Asante** - Data Scientist, Statistician and Financial Economist

Email: vkpoasante@gmail.com

## License
**MIT**
