import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- STEP 1: LOAD THE DATA ---
# This uses the local CSV file you've already downloaded
csv_file = 'FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv'

try:
    print(f"Loading data from {csv_file}...")
    df_raw = pd.read_csv(csv_file)
    print("Successfully loaded CSV.")
except FileNotFoundError:
    print("Error: The CSV file was not found in the directory!")
    # Optional: Keep your API code here as a backup if you wish

# --- STEP 2: CLEAN COLUMN NAMES ---
# This fixes the 'KeyError' by making all names lowercase and replacing spaces with underscores
df_raw.columns = [c.lower().replace(' ', '_') for c in df_raw.columns]

# --- STEP 3: CONVERT DATA TYPES ---
# Now 'expected_readmission_rate' will work because we cleaned the names in Step 2
df_raw['expected_readmission_rate'] = pd.to_numeric(df_raw['expected_readmission_rate'], errors='coerce')
df_raw['excess_readmission_ratio'] = pd.to_numeric(df_raw['excess_readmission_ratio'], errors='coerce')
df_raw['predicted_readmission_rate'] = pd.to_numeric(df_raw['predicted_readmission_rate'], errors='coerce')

# Drop rows that have N/A in our key columns
clean_df = df_raw.dropna(subset=['expected_readmission_rate', 'excess_readmission_ratio'])

print(f"Data ready for analysis. Number of clean records: {len(clean_df)}")


# --- 4. Objective 1.1: One-Way ANOVA ---
# Does ERR differ significantly across Measure Names?
anova_model = ols('excess_readmission_ratio ~ Q("measure_name")', data=clean_df).fit()
anova_results = sm.stats.anova_lm(anova_model, typ=2)
print("\n--- ANOVA Results ---")
print(anova_results)

# --- 5. Objective 1.2: Multiple Linear Regression ---
# Predicting ERR based on Expected Rate and Measure Name (categorical)
reg_model = ols('excess_readmission_ratio ~ expected_readmission_rate + Q("measure_name")', data=clean_df).fit()
print("\n--- Regression Model Summary ---")
print(reg_model.summary())

# --- 6. Visualizations (Deliverable C.6) ---
# Boxplot for ANOVA
plt.figure(figsize=(14, 8))
sns.boxplot(x='measure_name', y='excess_readmission_ratio', data=clean_df)
plt.title('Excess Readmission Ratio by Measure Type')
plt.xlabel('CMS Measure Name')
plt.ylabel('Excess Readmission Ratio (ERR)')
plt.xticks(rotation=45, ha = 'right')
plt.tight_layout()
plt.savefig('anova_boxplot.png')

# Scatter Plot for Regression
plt.figure(figsize=(10, 6))
sns.regplot(x='expected_readmission_rate', y='excess_readmission_ratio', 
            data=clean_df, scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
plt.title('Relationship: Expected Rate vs. Excess Readmission Ratio')
plt.tight_layout()
plt.savefig('regression_scatter.png')

print("\nAnalysis complete. Visualizations saved as 'anova_boxplot.png' and 'regression_scatter.png'.")
