import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load scaled data
df_scaled = pd.read_csv("../wine_data_scaled.csv")

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
plt.savefig("../figures/pca_scatter_plot.png")
plt.show()
