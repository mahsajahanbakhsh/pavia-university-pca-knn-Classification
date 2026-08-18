# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:01:14 2024

@author: Mahsa
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances  # Importing the missing function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from matplotlib.colors import ListedColormap

# 1. Load MATLAB image data
mat_file_image = 'D:/PhD/1/PR-dr lasanlou/Assignment-3/PaviaU.mat'  # path to your image
mat_file_gt = 'D:/PhD/1/PR-dr lasanlou/Assignment-3/PaviaU_gt.mat'  # path to your ground truth

# Load the .mat files
image_data = sio.loadmat(mat_file_image)
ground_truth_data = sio.loadmat(mat_file_gt)

# Extract the image bands and ground truth from the MATLAB files
image = image_data['paviaU']  # Adjust variable name if necessary
ground_truth = ground_truth_data['paviaU_gt']  # Adjust variable name if necessary

# Reshape the image bands to 2D array: (pixels, bands)
num_bands = image.shape[2]
num_pixels = image.shape[0] * image.shape[1]
image_reshaped = image.reshape((num_pixels, num_bands))

# 2. Calculate distance-based and similarity-based correlation matrices between image bands
# (i) Euclidean Distance (distance-based metric)
distance_matrix = pairwise_distances(image_reshaped.T, metric='euclidean')

# (ii) Cosine Similarity (similarity-based metric)
similarity_matrix = pairwise_distances(image_reshaped.T, metric='cosine')

# Plot correlation matrices
plt.figure(figsize=(12, 6))

# Plot distance matrix
plt.subplot(1, 2, 1)
sns.heatmap(distance_matrix, cmap='coolwarm')
plt.title('Euclidean Distance Matrix Between Bands')
plt.xlabel('Bands')
plt.ylabel('Bands')

# Plot similarity matrix
plt.subplot(1, 2, 2)
sns.heatmap(similarity_matrix, cmap='coolwarm')
plt.title('Cosine Similarity Matrix Between Bands')
plt.xlabel('Bands')
plt.ylabel('Bands')

plt.show()

# 3. KNN Classification using PCA for Feature Selection
X = image_reshaped
y = ground_truth.ravel()  # Flatten the ground truth labels to a 1D array

# Apply PCA for feature selection
n_components = 10  # Select top 10 components (you can change this)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# Explained variance ratio (to understand how much variance is preserved)
print(f"Explained variance ratio by PCA: {pca.explained_variance_ratio_}")

# Split the data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# KNN classification on the selected features (PCA components)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on the test data
y_pred = knn.predict(X_test)

# Calculate classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Classification Accuracy: {accuracy * 100:.2f}%")

# 4. Display the final classification map with class labels
# Define class labels (adjust to match your dataset's specific class labels)
class_labels = {
    0: 'Asphalt',
    1: 'Meadows',
    2: 'Gravel',
    3: 'Trees',
    4: 'Painted metal sheets',
    5: 'Bare Soil',
    6: 'Bitumen',
    7: 'Self-Blocking Bricks',
    8: 'Shadows'
}

# Create a color map for visualization with different colors for each class
cmap = ListedColormap(['gray', 'green', 'brown', 'darkgreen', 'red', 'tan', 'black', 'orange', 'blue'])

# Predict on the entire image data using PCA-reduced features
y_full_pred = knn.predict(X_pca)

# Reshape the classification result back to image shape
classified_image = y_full_pred.reshape((image.shape[0], image.shape[1]))

# Plot the classification map with labels
plt.figure(figsize=(10, 10))
plt.imshow(classified_image, cmap=cmap)  # Use the custom colormap
plt.title('Classified Map with Labels')
plt.colorbar(ticks=range(len(class_labels)), label='Classes')

# Create a legend with class labels
color_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) for i in range(len(class_labels))]
plt.legend(color_legend, class_labels.values(), loc='upper right', bbox_to_anchor=(1.25, 1))

plt.show()
