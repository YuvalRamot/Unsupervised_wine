import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Load scaled data
X_scaled = pd.read_csv("../wine_data_scaled.csv")

# Run t-SNE directly on scaled data (no PCA)
tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=500, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Plot t-SNE with KMeans labels
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8)
plt.title("t-SNE Projection of Wine Samples (Colored by K-Means Clusters)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.savefig("../figures/tsne_kmeans.png")
plt.show()
