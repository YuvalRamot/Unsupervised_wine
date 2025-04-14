import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Data preprocessing for wine dataset.")
    parser.add_argument("--input_path", type=str, default="../wine_data.csv", help="Path to the input raw data file.")
    parser.add_argument("--output_path", type=str, default="../wine_data_scaled.csv", help="Path to save the scaled data file.")
    args = parser.parse_args()

    # Use parsed arguments
    input_path = args.input_path
    output_path = args.output_path

    # Step 1: Load data
    print(f"📂 Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    # Step 2: Drop the 'quality' column
    X = df.drop(columns=['quality'])

    # Step 3: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 4: Save standardized data for future use
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df.to_csv(output_path, index=False)

    print(f"✅ Data preprocessing complete. Scaled data saved to: {output_path}")
