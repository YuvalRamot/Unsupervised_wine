import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import os

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="DBSCAN clustering on wine dataset.")
    parser.add_argument("--data_path", type=str, default="../wine_data_scaled.csv", help="Path to the input scaled data file.")
    parser.add_argument("--eps", type=float, default=0.17, help="Epsilon value for DBSCAN.")
    parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for DBSCAN.")
    parser.add_argument("--output_dir", type=str, default="../figures", help="Directory to save the output figures.")
    args = parser.parse_args()

    # Use parsed arguments
    data_path = args.data_path
    eps_value = args.eps
    min_samples = args.min_samples
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # === Step 1: Load and Reduce Data ===
    print(f"📂 Loading data from: {data_path}")
    df_scaled = pd.read_csv(data_path)

    # Reduce to 2D for clustering and visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)

    # === Step 2: Visualize K-Distance Graph ===
    print("📊 Step 1: Generating k-distance graph...")

    k = min_samples  # Same as min_samples
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.title("K-distance Graph (5th Nearest Neighbor)")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}th Nearest Neighbor Distance")
    plt.grid(True)
    plt.tight_layout()
    k_distance_path = os.path.join(output_dir, "dbscan_k_distance_explained.png")
    plt.savefig(k_distance_path)
    print(f"📁 K-distance graph saved to: {k_distance_path}")
    plt.show()

    print("📌 Choose eps just before the steep rise — typically around the elbow point.\n")

    # === Step 3: Apply DBSCAN with chosen eps ===
    print(f"🧪 Step 2: Applying DBSCAN with eps = {eps_value}, min_samples = {min_samples}...")

    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
    labels = dbscan.fit_predict(X_pca)

    # === Step 4: Evaluate the Result ===
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = np.sum(labels == -1)

    print(f"✅ Clusters found: {n_clusters}")
    print(f"❌ Outliers detected: {n_outliers}")

    # Silhouette score (excluding noise)
    if n_clusters > 1:
        mask = labels != -1
        sil_score = silhouette_score(X_pca[mask], labels[mask])
        print(f"📈 Silhouette Score (excluding outliers): {sil_score:.4f}")
    else:
        sil_score = -1
        print("⚠️ Not enough clusters for silhouette score.")

    # === Step 5: Visualize the Clusters ===
    print("🎨 Step 3: Visualizing and saving DBSCAN result...")

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = [plt.cm.tab10(i) if i != -1 else (0.2, 0.2, 0.2, 0.4) for i in unique_labels]

    for k, col in zip(unique_labels, colors):
        class_mask = (labels == k)
        plt.scatter(X_pca[class_mask, 0], X_pca[class_mask, 1],
                    s=15, c=[col], label=f'Cluster {k}' if k != -1 else 'Outliers', alpha=0.7)

    plt.title(f"DBSCAN Clustering (eps={eps_value})\nClusters: {n_clusters}, Outliers: {n_outliers}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(fontsize=9)
    plt.tight_layout()
    clustering_result_path = os.path.join(output_dir, "dbscan_result_explained.png")
    plt.savefig(clustering_result_path)
    print(f"📁 Clustering result saved to: {clustering_result_path}")
    plt.show()
