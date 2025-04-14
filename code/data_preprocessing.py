import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__=="__main__":

    # Step 1: Load data
    df = pd.read_csv("../wine_data.csv")

    # Step 2: Drop the 'quality' column
    X = df.drop(columns=['quality'])

    # Step 3: Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 4: Save standardized data for future use
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df.to_csv("../wine_data_scaled.csv", index=False)

    print("✅ Data preprocessing complete. Scaled data saved as 'wine_data_scaled.csv'")
