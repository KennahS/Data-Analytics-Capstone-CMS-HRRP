import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

def fetch_cms_data(dataset_id="9n3s-kdb3"):
    """Fetches all records from the CMS API using pagination."""
    base_url = f"https://data.cms.gov/data-api/v1/dataset/{dataset_id}/data"
    all_data = []
    offset = 0
    size = 1000  # CMS typically allows up to 5000 per request
    
    print("Fetching data from CMS API...")
    while True:
        params = {'size': size, 'offset': offset}
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            break
            
        data = response.json()
        if not data: # Break if no more data is returned
            break
            
        all_data.extend(data)
        offset += size
        print(f"Downloaded {len(all_data)} records...", end='\r')
        
        # For a quick test, you can uncomment the next line:
        # if offset >= 2000: break 

    print(f"\nSuccessfully downloaded {len(all_data)} records.")
    return pd.DataFrame(all_data)

# --- 1. Data Collection ---
df_raw = fetch_cms_data()

# --- 2. Data Cleaning ---
# Convert relevant columns to numeric, removing 'N/A' strings
df_raw['expected_readmission_rate'] = pd.to_numeric(df_raw['expected_readmission_rate'], errors='coerce')
df_raw['excess_readmission_ratio'] = pd.to_numeric(df_raw['excess_readmission_ratio'], errors='coerce')

# Drop rows missing our key metrics
clean_df = df_raw.dropna(subset=['expected_readmission_rate', 'excess_readmission_ratio'])

# --- 3. Objective 1.1: One-Way ANOVA ---
# Does ERR differ significantly across Measure Names?
anova_model = ols('excess_readmission_ratio ~ Q("measure_name")', data=clean_df).fit()
anova_results = sm.stats.anova_lm(anova_model, typ=2)
print("\n--- ANOVA Results ---")
print(anova_results)

# --- 4. Objective 1.2: Multiple Linear Regression ---
# Predicting ERR based on Expected Rate and Measure Name (categorical)
reg_model = ols('excess_readmission_ratio ~ expected_readmission_rate + Q("measure_name")', data=clean_df).fit()
print("\n--- Regression Model Summary ---")
print(reg_model.summary())

# --- 5. Visualizations (Deliverable C.6) ---
# Boxplot for ANOVA
plt.figure(figsize=(12, 6))
sns.boxplot(x='measure_name', y='excess_readmission_ratio', data=clean_df)
plt.title('Excess Readmission Ratio by Measure Type')
plt.xticks(rotation=45)
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
