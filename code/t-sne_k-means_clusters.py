import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="t-SNE and K-Means clustering on wine dataset.")
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of clusters for K-Means.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="Perplexity for t-SNE.")
    parser.add_argument("--n_iter", type=int, default=500, help="Number of iterations for t-SNE.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--data_path", type=str, default="../wine_data_scaled.csv", help="Path to the input scaled data file.")
    parser.add_argument("--output_dir", type=str, default="../figures", help="Directory to save the output figures.")
    args = parser.parse_args()

    # Use parsed arguments
    random_state = args.seed
    data_path = args.data_path
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load scaled data
    print(f"📂 Loading data from: {data_path}")
    X_scaled = pd.read_csv(data_path)

    # Run t-SNE directly on scaled data (no PCA)
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init='pca', n_iter=args.n_iter, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)

    # KMeans clustering
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)

    # Plot t-SNE with KMeans labels
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8)
    plt.title("t-SNE Projection of Wine Samples (Colored by K-Means Clusters)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "tsne_kmeans.png")
    plt.savefig(output_path)
    print(f"📁 Figure saved to: {output_path}")
    plt.show()
