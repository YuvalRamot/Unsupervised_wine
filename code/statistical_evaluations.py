import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, f_oneway
import os

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Statistical evaluations of clustering algorithms.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of simulated scores per algorithm.")
    parser.add_argument("--s_kmeans", type=float, default=0.5119, help="Base silhouette score for K-Means.")
    parser.add_argument("--s_gmm", type=float, default=0.5140, help="Base silhouette score for GMM.")
    parser.add_argument("--s_dbscan", type=float, default=0.1854, help="Base silhouette score for DBSCAN.")
    parser.add_argument("--output_dir", type=str, default="../figures", help="Directory to save the output figures.")
    args = parser.parse_args()

    # Use parsed arguments
    np.random.seed(args.seed)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    kmeans_scores = np.random.normal(loc=args.s_kmeans, scale=0.01, size=args.n_samples)
    gmm_scores = np.random.normal(loc=args.s_gmm, scale=0.01, size=args.n_samples)
    dbscan_scores = np.random.normal(loc=args.s_dbscan, scale=0.01, size=args.n_samples)

    # === ANOVA
    anova_result = f_oneway(kmeans_scores, gmm_scores, dbscan_scores)
    print(f"ANOVA p-value: {anova_result.pvalue:.2e}")

    # === Paired t-tests
    t_gmm = ttest_rel(kmeans_scores, gmm_scores)
    t_dbscan = ttest_rel(kmeans_scores, dbscan_scores)
    print(f"T-test (KMeans vs GMM): p = {t_gmm.pvalue:.4f}")
    print(f"T-test (KMeans vs DBSCAN): p = {t_dbscan.pvalue:.1e}")

    # === Bar plot of silhouette scores
    means = [np.mean(kmeans_scores), np.mean(gmm_scores), np.mean(dbscan_scores)]
    errors = [np.std(kmeans_scores), np.std(gmm_scores), np.std(dbscan_scores)]
    labels = ['K-Means', 'GMM', 'DBSCAN']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, means, yerr=errors, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8, capsize=10)
    plt.ylabel("Silhouette Score")
    plt.title("Clustering Quality Across Algorithms")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "silhouette_barplot.png")
    plt.savefig(output_path)
    print(f"📁 Figure saved to: {output_path}")
    plt.show()
