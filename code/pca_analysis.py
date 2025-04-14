import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the scaled data
df_scaled = pd.read_csv("../wine_data_scaled.csv")

# Step 2: Apply PCA
pca = PCA()
X_pca = pca.fit_transform(df_scaled)

# Step 3: Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()

# Step 4: Save figure to /figures/ folder
plt.savefig("../figures/pca_explained_variance.png")
plt.show()
