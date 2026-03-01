import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load the Excel file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "EVP-3 COOLING TOWER (2).xlsx")

# Use header=1 to skip the first empty/informational row
# Adjust header= if your column names start on a different row
df = pd.read_excel(file_path, header=1)

# 2. Clean column names to ensure they match exactly
df.columns = df.columns.str.strip().str.upper()

# 3. Clean and define features
# Ensure your columns in the list below match your Excel headers exactly
features = ['HOT WATER TEMPERATURE', 'COLD WATER TEMPERATURE', 'WBT', 'CIRCULATING WATER', 'COC']
df_clean = df.dropna(subset=features + ['EFFICIENCY', 'BLOW DOWN LOSS', 'WATER MAKE UP'])

X = df_clean[features]

# 4. Train and save
# Mapping internal names to your specific Excel column headers
targets = {
    "efficiency": 'EFFICIENCY', 
    "blowdown": 'BLOW DOWN LOSS', 
    "makeup": 'WATER MAKE UP'
}

for name, col_name in targets.items():
    model = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
    model.fit(X, df_clean[col_name])
    with open(f"model_{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model {name} saved successfully.")