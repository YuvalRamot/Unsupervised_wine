import argparse
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="PCA scatter plot for wine dataset.")
    parser.add_argument("--data_path", type=str, default="../wine_data_scaled.csv", help="Path to the input scaled data file.")
    parser.add_argument("--output_dir", type=str, default="../figures", help="Directory to save the output figure.")
    args = parser.parse_args()

    # Use parsed arguments
    data_path = args.data_path
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load scaled data
    print(f"📂 Loading data from: {data_path}")
    df_scaled = pd.read_csv(data_path)

    # Apply PCA and reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(df_scaled)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.4, s=15)
    plt.title("Wine Samples in 2D PCA Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, "pca_scatter_plot.png")
    plt.savefig(output_path)
    print(f"📁 Figure saved to: {output_path}")
    plt.show()
