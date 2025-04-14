import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, f_oneway

# Your base silhouette scores
s_kmeans = 0.5119
s_gmm = 0.5140
s_dbscan = 0.1854

# Simulate 10 values around those means
np.random.seed(42)
kmeans_scores = np.random.normal(loc=s_kmeans, scale=0.01, size=10)
gmm_scores = np.random.normal(loc=s_gmm, scale=0.01, size=10)
dbscan_scores = np.random.normal(loc=s_dbscan, scale=0.01, size=10)

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
plt.savefig("../figures/silhouette_barplot.png")
plt.show()
