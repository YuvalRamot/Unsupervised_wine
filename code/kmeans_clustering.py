import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === Step 1: Load and Reduce Data ===
df_scaled = pd.read_csv("../wine_data_scaled.csv")

# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)

# === Step 2: Elbow Method to Choose K ===
print("📊 Step 1: Running elbow method to find optimal K...")

output_dir = "../figures/kmeans_each_K"
os.makedirs(output_dir, exist_ok=True)

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    inertias.append(kmeans.inertia_)

    # Save cluster plot for each K
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=15, alpha=0.6)
    plt.title(f"K-Means Clustering (K={k})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kmeans_clusters_k{k}.png")
    plt.close()

# Save and show the Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.title("Elbow Method for Optimal K (K-Means)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/kmeans_elbow.png")
plt.show()

# === Step 3: Apply KMeans with Chosen K ===
optimal_k = 3  # Based on elbow
print(f"✅ Step 2: Applying KMeans with K={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
labels_final = kmeans_final.fit_predict(X_pca)

# Calculate silhouette score
sil_score = silhouette_score(X_pca, labels_final)
print(f"📈 Silhouette Score (KMeans, K={optimal_k}): {sil_score:.4f}")

# === Step 4: Visualize and Save Final Clustering ===
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_final, cmap='tab10', s=15, alpha=0.6)
plt.title(f"K-Means Clustering Result (K={optimal_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/kmeans_clustered_pca_k3.png")
plt.show()

# === Step 5: Feature Summary per Cluster ===
print("\n📊 Feature Summary per Cluster (K-Means)...")
df_scaled_full = pd.read_csv("../wine_data_scaled.csv")  # Reload full scaled features
df_scaled_full['cluster'] = labels_final

# Define features to show
features_to_show = ['alcohol', 'residual_sugar', 'fixed_acidity', 'pH', 'sulphates']

# Calculate and display the means
summary = df_scaled_full.groupby('cluster')[features_to_show].mean()
print(summary.round(3))
