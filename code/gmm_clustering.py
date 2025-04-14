import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# === Step 1: Load data and apply PCA ===
df_scaled = pd.read_csv("../wine_data_scaled.csv")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)

# === Step 2: Fit GMM ===
n_components = 3  # same as KMeans
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(X_pca)
labels = gmm.predict(X_pca)

# === Step 3: Silhouette Score ===
sil_score = silhouette_score(X_pca, labels)
print(f"📈 Silhouette Score (GMM, K=3): {sil_score:.4f}")

# === Step 4: Visualization ===
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=15, alpha=0.6)
plt.title(f"GMM Clustering Result (K={n_components})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/gmm_clustered_pca_k3.png")
plt.show()

# === Step 5: Anomaly Detection Using GMM Log-Likelihood ===
print("\n🔍 Step 5: GMM Anomaly Detection")

# Score each point with log-likelihood under the fitted GMM
log_likelihoods = gmm.score_samples(X_pca)

# Define anomaly threshold (3 std dev below mean)
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
plt.savefig("../figures/gmm_anomaly_score_hist.png")
plt.show()
# === Step 7: Visualize Anomalies in PCA Space ===
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[~anomalies, 0], X_pca[~anomalies, 1],
            s=15, alpha=0.5, label='Normal', c='gray')
plt.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1],
            s=25, alpha=0.9, label='Anomaly', c='red')
plt.title("GMM Anomaly Detection in PCA Space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/gmm_anomaly_scatter.png")
plt.show()

