# Unsupervised Wine Project

This project applies various unsupervised learning techniques to analyze and cluster a wine dataset. The project includes data preprocessing, dimensionality reduction, clustering, and statistical evaluations.

## Prerequisites

Make sure you have Python installed (version 3.7 or higher). Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Project Structure

- **`data_preprocessing.py`**: Preprocesses the raw wine dataset by standardizing the features.
- **`pca_analysis.py`**: Performs PCA and plots the cumulative explained variance.
- **`pca_scatter.py`**: Visualizes the wine dataset in 2D PCA space.
- **`kmeans_clustering.py`**: Applies K-Means clustering and visualizes the results.
- **`gmm_clustering.py`**: Applies Gaussian Mixture Model (GMM) clustering and performs anomaly detection.
- **`dbscan_clustering.py`**: Applies DBSCAN clustering and visualizes the results.
- **`t-sne_k-means_clusters.py`**: Combines t-SNE for dimensionality reduction with K-Means clustering.
- **`statistical_evaluations.py`**: Performs statistical evaluations of clustering algorithms.

## How to Run the Scripts

### 1. Data Preprocessing
Preprocess the raw wine dataset by standardizing the features.

```bash
python code/data_preprocessing.py --input_path "path/to/wine_data.csv" --output_path "path/to/wine_data_scaled.csv"
```

### 2. PCA Analysis
Perform PCA and plot the cumulative explained variance.

```bash
python code/pca_analysis.py --data_path "path/to/wine_data_scaled.csv" --output_dir "path/to/output"
```

### 3. PCA Scatter Plot
Visualize the wine dataset in 2D PCA space.

```bash
python code/pca_scatter.py --data_path "path/to/wine_data_scaled.csv" --output_dir "path/to/output"
```

### 4. K-Means Clustering
Apply K-Means clustering and visualize the results.

```bash
python code/kmeans_clustering.py --data_path "path/to/wine_data_scaled.csv" --output_dir "path/to/output" --k_range 2 10 --optimal_k 3
```

### 5. GMM Clustering
Apply Gaussian Mixture Model (GMM) clustering and perform anomaly detection.

```bash
python code/gmm_clustering.py --data_path "path/to/wine_data_scaled.csv" --output_dir "path/to/output" --n_components 3
```

### 6. DBSCAN Clustering
Apply DBSCAN clustering and visualize the results.

```bash
python code/dbscan_clustering.py --data_path "path/to/wine_data_scaled.csv" --output_dir "path/to/output" --eps 0.2 --min_samples 5
```

### 7. t-SNE with K-Means Clustering
Combine t-SNE for dimensionality reduction with K-Means clustering.

```bash
python code/t-sne_k-means_clusters.py --data_path "path/to/wine_data_scaled.csv" --output_dir "path/to/output" --n_clusters 3 --perplexity 30 --n_iter 500
```

### 8. Statistical Evaluations
Perform statistical evaluations of clustering algorithms.

```bash
python code/statistical_evaluations.py --output_dir "path/to/output" --n_samples 10 --s_kmeans 0.5119 --s_gmm 0.5140 --s_dbscan 0.1854
```

## Notes
- Replace `"path/to/..."` with the actual paths to your input data and desired output directories.
- The default paths for input and output files are specified in each script. If no arguments are provided, the scripts will use these defaults.

## License
This project is for educational purposes and is licensed under the MIT License.