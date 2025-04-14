import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="GMM clustering on wine dataset.")
    parser.add_argument("--data_path", type=str, default="../wine_data_scaled.csv", help="Path to the input scaled data file.")
    parser.add_argument("--n_components", type=int, default=3, help="Number of components for GMM.")
    parser.add_argument("--output_dir", type=str, default="../figures", help="Directory to save the output figures.")
    args = parser.parse_args()

    # Use parsed arguments
    data_path = args.data_path
    n_components = args.n_components
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # === Step 1: Load data and apply PCA ===
    print(f"📂 Loading data from: {data_path}")
    df_scaled = pd.read_csv(data_path)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)

    # === Step 2: Fit GMM ===
    print(f"📊 Fitting GMM with {n_components} components...")
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_pca)
    labels = gmm.predict(X_pca)

    # === Step 3: Silhouette Score ===
    sil_score = silhouette_score(X_pca, labels)
    print(f"📈 Silhouette Score (GMM, K={n_components}): {sil_score:.4f}")

    # === Step 4: Visualization ===
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=15, alpha=0.6)
    plt.title(f"GMM Clustering Result (K={n_components})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    clustering_plot_path = os.path.join(output_dir, f"gmm_clustered_pca_k{n_components}.png")
    plt.savefig(clustering_plot_path)
    print(f"📁 Clustering plot saved to: {clustering_plot_path}")
    plt.show()

    # === Step 5: Anomaly Detection Using GMM Log-Likelihood ===
    print("\n🔍 Step 5: GMM Anomaly Detection")
    log_likelihoods = gmm.score_samples(X_pca)
    mu = np.mean(log_likelihoods)
    sigma = np.std(log_likelihoods)
    threshold = mu - 3 * sigma
    anomalies = log_likelihoods < threshold

    print(f"📉 Anomaly threshold: {threshold:.2f}")
    print(f"❗ Number of GMM anomalies: {np.sum(anomalies)} out of {len(X_pca)} samples")

    # === Step 6: Plot Histogram of Scores ===
    plt.figure(figsize=(8, 5))
    plt.hist(log_likelihoods, bins=50, color='gray', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.title("GMM Log-Likelihood Scores for Wine Samples")
    plt.xlabel("Log-Likelihood")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    histogram_plot_path = os.path.join(output_dir, "gmm_anomaly_score_hist.png")
    plt.savefig(histogram_plot_path)
    print(f"📁 Histogram plot saved to: {histogram_plot_path}")
    plt.show()

    # === Step 7: Visualize Anomalies in PCA Space ===
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1], s=15, alpha=0.5, label='Normal', c='gray')
    plt.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], s=25, alpha=0.9, label='Anomaly', c='red')
    plt.title("GMM Anomaly Detection in PCA Space")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()
    anomaly_plot_path = os.path.join(output_dir, "gmm_anomaly_scatter.png")
    plt.savefig(anomaly_plot_path)
    print(f"📁 Anomaly scatter plot saved to: {anomaly_plot_path}")
    plt.show()

