import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Config
manifest_input='total_mnist.csv'#'manifest.csv'
output_dir='output'

train_proportion=0.8
val_proportion=0.1
test_proportion=0.1

assert train_proportion+val_proportion+test_proportion==1
# Get paths
os.makedirs(output_dir, exist_ok=True)

manifest_input_path=os.path.join('data',manifest_input)
manifest_train_path=os.path.join('output','train_'+manifest_input)
manifest_val_path=os.path.join('output','val_'+manifest_input)
manifest_test_path=os.path.join('output','test_'+manifest_input)

# Load the original manifest.csv
df = pd.read_csv(manifest_input_path)

# Split the data into train, validation, and test sets
train_df, val_test_df = train_test_split(df, test_size=val_proportion+test_proportion, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=test_proportion/(1-train_proportion), random_state=42)

# Save the split data to CSV files
train_df.to_csv(manifest_train_path, index=False)
val_df.to_csv(manifest_val_path, index=False)
test_df.to_csv(manifest_test_path, index=False)

print("Data split and saved successfully.")
