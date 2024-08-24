import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('shows.csv')

# Select features for PCA 
features = ['runtime', 'seasons', 'imdb_votes', 'tmdb_popularity', 'age_certification']

# Extract the selected features
selectedFeatures = data[features]

# Standardize the features
scaler = StandardScaler()
selectedFeatures_scaled = scaler.fit_transform(selectedFeatures)

# Perform PCA
pca = PCA()
selectedFeatures_pca = pca.fit_transform(selectedFeatures_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Scree plot (explained variance)
plt.figure(figsize=(12, 5))

# Plot 1: Scree Plot - Explained Variance Ratio
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot - Explained Variance Ratio (Shows)')

# Plot 2: Cumulative Explained Variance Plot
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot (Shows)')
plt.grid(True)

plt.tight_layout()
plt.savefig('pca_variance_plots_shows.png')
plt.close()

# Visualize Feature Loadings in Principal Components
plt.figure(figsize=(10, 6))
for i in range(len(features)):
    plt.scatter(selectedFeatures_pca[:, 0], selectedFeatures_pca[:, 1], c=selectedFeatures_scaled[:, i], cmap='viridis', marker='o', label=features[i])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Feature Loadings in Principal Components (Shows)')
plt.colorbar(label='Feature Value')
plt.legend()
plt.grid(True)
plt.savefig('pca_feature_loadings_shows.png')
plt.close()

# Output explained variance ratios for each principal component
print("Explained Variance Ratios for Principal Components:")
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f"PC{i}: {ratio:.4f}")

# Output cumulative explained variance for each number of components
print("\nCumulative Explained Variance:")
for i, variance in enumerate(cumulative_variance, 1):
    print(f"Components {i}: {variance:.4f}")

# Output the loadings of original features in the first two principal components
print("\nLoadings of Original Features in First Two Principal Components:")
for i, feature in enumerate(features):
    print(f"{feature} in PC1: {pca.components_[0, i]:.4f}, PC2: {pca.components_[1, i]:.4f}")
