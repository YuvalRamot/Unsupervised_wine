import argparse
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="PCA analysis for wine dataset.")
    parser.add_argument("--data_path", type=str, default="../wine_data_scaled.csv", help="Path to the input scaled data file.")
    parser.add_argument("--output_dir", type=str, default="../figures", help="Directory to save the output figure.")
    args = parser.parse_args()

    # Use parsed arguments
    data_path = args.data_path
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load the scaled data
    print(f"📂 Loading data from: {data_path}")
    df_scaled = pd.read_csv(data_path)

    # Step 2: Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(df_scaled)

    # Step 3: Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()

    # Step 4: Save figure to the specified output directory
    output_path = os.path.join(output_dir, "pca_explained_variance.png")
    plt.savefig(output_path)
    print(f"📁 Figure saved to: {output_path}")
    plt.show()
