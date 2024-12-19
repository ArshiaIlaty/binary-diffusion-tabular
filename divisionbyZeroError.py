import pandas as pd

df = pd.read_csv("./data/cervical_train.csv")

# Check for columns where all values are the same
print("Columns with constant values:")
for col in df.columns:
    if df[col].nunique() == 1:
        print(f"{col}: unique value = {df[col].iloc[0]}")

# Also check for potential numerical issues
print("\nValue ranges for each column:")
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        print(f"{col}: min = {df[col].min()}, max = {df[col].max()}")
