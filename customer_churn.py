# Auto-exported from Customer_Churn.ipynb (run from the folder containing Telco_Churn_Data.csv)
# # Predicting Customer Churn in a Telecom Company
# 
# This notebook cleans and explores a telecom customer dataset, then trains baseline classifiers to predict churn.
# 
# **Contents**
# 1. Importing libraries and loading the data
# 2. Shape and structure of the data
# 3. Renaming columns
# 4. Descriptive statistics
# 5. Fixing data types
# 6. Missing values
# 7. Data quality checks
# 8. Exploring the target variable
# 9. Correlation analysis
# 10. Univariate and bivariate analysis
# 11. Predictive modelling
# 12. Conclusions

# ## 1. Importing libraries and loading the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, classification_report)

sns.set_style('whitegrid')
RANDOM_STATE = 42

df = pd.read_csv('Telco_Churn_Data.csv')
df.head()

df.info()

# ## 2. Checking the length and shape of the data

print(f'Rows: {len(df):,} | Columns: {df.shape[1]}')

# ## 3. Renaming columns in a readable format
# 
# The rename mapping is checked against the actual column names first, so a typo can no longer fail silently.

rename_map = {
    'Target Churn': 'Target_Churn',
    'Target Code': 'Target_Code',
    'Current Bill Amt': 'Current_Bill_Amt',
    'Avg Calls': 'Avg_Calls',
    'Avg Calls Weekdays': 'Avg_Calls_Weekdays',
    'Account Age': 'Account_Age',
    'Percent Increase MOM': 'Percent_Increase_MOM',
    'Acct Plan Subtype': 'Acct_Plan_Subtype',
    'Complaint Code': 'Complaint_Code',
    'Avg Days Delinquent': 'Avg_Days_Delinquent',
    'Current TechSupComplaints': 'Current_TechSupComplaints',
    'Current Days OpenWorkOrders': 'Current_Days_OpenWorkOrders',
    'Equipment Age': 'Equipment_Age',
    'Condition of Current Handset': 'Condition_of_Current_Handset',
    'Avg Hours WorkOrderOpenned': 'Avg_Hours_WorkOrderOpened',  # source column has a typo ("Openned")
}

missing_keys = set(rename_map) - set(df.columns)
assert not missing_keys, f'Columns not found in the data: {missing_keys}'

df = df.rename(columns=rename_map)
df.head()

# ## 4. Description of the data

df.describe().T

# ## 5. Fixing data types
# 
# - `Condition_of_Current_Handset` is a code (1-3), so it is treated as categorical.
# - `Current_TechSupComplaints` is a count and `Target_Code` is a 0/1 label; both stay numeric.

df['Condition_of_Current_Handset'] = df['Condition_of_Current_Handset'].astype('category')
df[['Target_Code', 'Condition_of_Current_Handset', 'Current_TechSupComplaints']].dtypes

# ## 6. Missing values
# 
# Categorical gaps are filled with the mode for exploration. For modelling (section 11) imputation is done inside the pipeline and fitted on the training data only, to avoid leakage.

# Keep an un-imputed copy for the modelling pipeline
df_model_raw = df.copy()
df.isnull().sum()[df.isnull().sum() > 0]

df['Complaint_Code'] = df['Complaint_Code'].fillna(df['Complaint_Code'].mode()[0])
df['Condition_of_Current_Handset'] = df['Condition_of_Current_Handset'].fillna(
    df['Condition_of_Current_Handset'].mode()[0])

print('Remaining missing values:', int(df.isnull().sum().sum()))

# ## 7. Data quality checks

print('Duplicate rows:', df.duplicated().sum())
print('Negative bill amounts:', (df['Current_Bill_Amt'] < 0).sum())
print(f"Share of rows where Avg_Calls_Weekdays > Avg_Calls: "
      f"{(df['Avg_Calls_Weekdays'] > df['Avg_Calls']).mean():.1%}")

# **Notes**
# - A few bills are negative (possibly credits/refunds) and several fields are heavily right-skewed. They are kept here, but should be confirmed with the data owner.
# - Weekday calls exceed total average calls in most rows, which is unexpected for a subset of the same calls. The definitions of these two fields should be confirmed.

skewed = ['Current_Bill_Amt', 'Avg_Calls', 'Avg_Calls_Weekdays', 'Percent_Increase_MOM']
fig, axes = plt.subplots(1, len(skewed), figsize=(16, 4))
for ax, col in zip(axes, skewed):
    sns.boxplot(y=df[col], ax=ax, color='lightgray')
    ax.set_title(col)
plt.tight_layout()
plt.show()

# ## 8. Exploration of the Target_Churn variable

print(df['Target_Churn'].value_counts())
print()
print(df['Target_Churn'].value_counts(normalize=True).round(3))

sns.countplot(data=df, x='Target_Churn', hue='Target_Churn', palette=['dimgray', 'black'], order=['No Churn', 'Churn'], legend=False)
plt.title('Churn vs No Churn')
plt.ylabel('No of Customers')
plt.show()

df.groupby('Target_Churn', observed=True).mean(numeric_only=True).T

# ## 9. Correlation matrix of variables
# 
# `Target_Code` (0/1) is included, so its correlations are point-biserial correlations with churn.

numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr().round(2)
corr_matrix

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Variables')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

corr_matrix['Target_Code'].drop('Target_Code').sort_values(key=abs, ascending=False)

# **Observations**
# - `Avg_Calls` and `Avg_Calls_Weekdays` have the strongest correlation (r = 0.73). Using both in a linear model risks multicollinearity.
# - `Avg_Days_Delinquent` has the strongest relationship with churn (r = 0.46).
# - `Avg_Calls_Weekdays` and `Current_Bill_Amt` are moderately related (r = 0.43); `Avg_Calls` and `Current_Bill_Amt` less so (r = 0.35).
# - `Current_TechSupComplaints` and `Avg_Hours_WorkOrderOpened` are weakly related (r = 0.23).
# - All other pairs have very weak linear correlation (|r| < 0.15).

# ## 10. Univariate and bivariate analyses

# ### 10.1 Univariate analysis: Avg_Calls_Weekdays, Avg_Calls and Current_Bill_Amt

columns = ['Avg_Calls_Weekdays', 'Avg_Calls', 'Current_Bill_Amt']
df[columns].describe()

for col in columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('No of Customers')
    plt.tight_layout()
    plt.show()

# ### 10.2 Categorical variables versus churn (churn rate per category)

def plot_churn_rate(col, min_count=1):
    """Bar chart of the churn rate per category, annotated with group size."""
    stats = (df.groupby(col, observed=True)['Target_Code']
               .agg(churn_rate='mean', n='size')
               .query('n >= @min_count')
               .sort_values('churn_rate'))
    ax = stats['churn_rate'].plot.barh(figsize=(9, 5), color='dimgray')
    for i, (rate, n) in enumerate(zip(stats['churn_rate'], stats['n'])):
        ax.text(rate + 0.01, i, f'{rate:.0%}  (n={n})', va='center')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Churn rate')
    ax.set_title(f'Churn rate by {col}')
    plt.show()
    return stats

plot_churn_rate('Complaint_Code')

plot_churn_rate('Acct_Plan_Subtype')

plot_churn_rate('Current_TechSupComplaints')

plot_churn_rate('Condition_of_Current_Handset')

# Counts matter as well as rates: several complaint codes have very few customers (for example `Pricing` and `Inaccurate Sales Inf`), so their rates are unreliable.

# ### 10.3 Numeric variables versus churn

def plot_kde_by_churn(col, xlabel=None):
    plt.figure(figsize=(12, 5))
    sns.kdeplot(data=df, x=col, hue='Target_Churn', hue_order=['No Churn', 'Churn'],
                fill=True, common_norm=False, palette=['dimgray', 'black'],
                alpha=0.5, linewidth=1.5)
    plt.title(f'{col} - Churn vs No Churn')
    plt.xlabel(xlabel or col)
    plt.ylabel('Density')
    plt.show()

plot_kde_by_churn('Avg_Days_Delinquent', 'Average number of days delinquent')

plot_kde_by_churn('Account_Age')

# Percent_Increase_MOM has extreme outliers; clip the plot to the 1st-99th percentile
low, high = df['Percent_Increase_MOM'].quantile([0.01, 0.99])
df_plot = df[df['Percent_Increase_MOM'].between(low, high)]

plt.figure(figsize=(12, 5))
sns.kdeplot(data=df_plot, x='Percent_Increase_MOM', hue='Target_Churn',
            hue_order=['No Churn', 'Churn'], fill=True, common_norm=False,
            palette=['dimgray', 'black'], alpha=0.5, linewidth=1.5)
plt.title('Percent_Increase_MOM (1st-99th percentile) - Churn vs No Churn')
plt.xlabel('Percent_Increase_MOM')
plt.ylabel('Density')
plt.show()

# ## 11. Predictive modelling
# 
# **Setup**
# - Target: `Target_Code` (1 = churn). `Target_Churn` is the same label as text, so it is dropped from the features to prevent leakage.
# - Stratified 80/20 train/test split.
# - Preprocessing is inside a pipeline: median imputation + scaling for numeric columns, mode imputation + one-hot encoding for categorical columns. Everything is fitted on the training data only.

X = df_model_raw.drop(columns=['Target_Churn', 'Target_Code'])
y = df_model_raw['Target_Code']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

categorical_features = ['Acct_Plan_Subtype', 'Complaint_Code', 'Condition_of_Current_Handset']
numeric_features = [c for c in X.columns if c not in categorical_features]

# Handset code is numeric in the raw data; treat it as a category explicitly
for part in (X_train, X_test):
    part['Condition_of_Current_Handset'] = part['Condition_of_Current_Handset'].astype(object)

numeric_pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('scale', StandardScaler())])
categorical_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                             ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocess = ColumnTransformer([('num', numeric_pipe, numeric_features),
                                ('cat', categorical_pipe, categorical_features)])

print(f'Train: {X_train.shape} | Test: {X_test.shape}')
print(f'Churn rate - train: {y_train.mean():.3f} | test: {y_test.mean():.3f}')

# ### 11.1 Train and compare models

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
}

fitted, rows = {}, []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    pipe = Pipeline([('prep', preprocess), ('model', model)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    fitted[name] = pipe
    rows.append({'Model': name,
                 'CV ROC-AUC (train)': cv_auc,
                 'Accuracy': accuracy_score(y_test, pred),
                 'Precision': precision_score(y_test, pred),
                 'Recall': recall_score(y_test, pred),
                 'F1': f1_score(y_test, pred),
                 'Test ROC-AUC': roc_auc_score(y_test, proba)})

results = pd.DataFrame(rows).set_index('Model').round(3)
results

# ### 11.2 Evaluate the best model on the test set

best_name = results['Test ROC-AUC'].idxmax()
best = fitted[best_name]
print(f'Best model by test ROC-AUC: {best_name}\n')
print(classification_report(y_test, best.predict(X_test), target_names=['No Churn', 'Churn']))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_estimator(best, X_test, y_test,
                                      display_labels=['No Churn', 'Churn'],
                                      cmap='Greys', ax=axes[0])
axes[0].set_title(f'Confusion matrix - {best_name}')
axes[0].grid(False)

for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=axes[1])
axes[1].plot([0, 1], [0, 1], 'k--', label='Chance')
axes[1].set_title('ROC curves')
axes[1].legend()

plt.tight_layout()
plt.show()

# ### 11.3 Feature importance

feature_names = best.named_steps['prep'].get_feature_names_out()
model = best.named_steps['model']

if hasattr(model, 'feature_importances_'):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    label = 'Importance'
else:  # logistic regression: use absolute coefficients (features are scaled)
    importance = pd.Series(np.abs(model.coef_[0]), index=feature_names)
    label = 'Absolute coefficient'

importance = importance.sort_values().tail(15)
importance.plot.barh(figsize=(9, 6), color='dimgray')
plt.xlabel(label)
plt.title(f'Top 15 features - {best_name}')
plt.tight_layout()
plt.show()

# ## 12. Conclusions

# - The classes are roughly balanced (about 49% churn), so accuracy is meaningful, but precision, recall and ROC-AUC give a fuller picture.
# - `Avg_Days_Delinquent` is the clearest driver of churn; most other numeric variables are only weakly related to it in a linear sense.
# - The call and bill variables are correlated with one another, so linear models should be interpreted with care.
# - In the run documented here, tree-based models beat logistic regression (Random Forest test ROC-AUC ~0.90 vs ~0.86 for Logistic Regression), suggesting non-linear effects. Exact numbers may vary slightly with library versions.
# - `Avg_Days_Delinquent` and `Percent_Increase_MOM` carry the most importance in the tree model. `Percent_Increase_MOM` has almost no linear correlation with churn, so its importance points to a non-linear pattern worth investigating (for example, the many zero values).
# 
# **Possible next steps**
# - Clarify field definitions and units with the data owner (calls, bills, delinquency), and review outliers and negative bills.
# - Tune hyperparameters (for example with `GridSearchCV`) and test a probability threshold that matches the business cost of missing a churner.
# - Try log-transforming or winsorizing skewed variables for the linear model.

