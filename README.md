# Unsupervised Wine Project

This project applies various unsupervised learning techniques to analyze and cluster a wine dataset. The project includes data preprocessing, dimensionality reduction, clustering, and statistical evaluations.

## Prerequisites

```bash
pip install -r requirements.txt


Project Structure
data_preprocessing.py: Preprocesses the raw wine dataset by standardizing the features.
pca_analysis.py: Performs PCA and plots the cumulative explained variance.
pca_scatter.py: Visualizes the wine dataset in 2D PCA space.
kmeans_clustering.py: Applies K-Means clustering and visualizes the results.
gmm_clustering.py: Applies Gaussian Mixture Model (GMM) clustering and performs anomaly detection.
dbscan_clustering.py: Applies DBSCAN clustering and visualizes the results.
t-sne_k-means_clusters.py: Combines t-SNE for dimensionality reduction with K-Means clustering.
statistical_evaluations.py: Performs statistical evaluations of clustering algorithms.
How to Run the Scripts
1. Data Preprocessing
Preprocess the raw wine dataset by standardizing the features.

2. PCA Analysis
Perform PCA and plot the cumulative explained variance.

3. PCA Scatter Plot
Visualize the wine dataset in 2D PCA space.

4. K-Means Clustering
Apply K-Means clustering and visualize the results.

5. GMM Clustering
Apply Gaussian Mixture Model (GMM) clustering and perform anomaly detection.

6. DBSCAN Clustering
Apply DBSCAN clustering and visualize the results.

7. t-SNE with K-Means Clustering
Combine t-SNE for dimensionality reduction with K-Means clustering.

8. Statistical Evaluations
Perform statistical evaluations of clustering algorithms.

Notes
Replace "path/to/..." with the actual paths to your input data and desired output directories.
The default paths for input and output files are specified in each script. If no arguments are provided, the scripts will use these defaults.
License
This project is for educational purposes and is licensed under the MIT License.