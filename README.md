# pavia-university-pca-knn-Classification
# Pavia University Hyperspectral Classification Using PCA and KNN

## Overview

This project investigates hyperspectral image classification using the **Pavia University scene** and **Principal Component Analysis (PCA)** for dimensionality reduction, followed by **K-Nearest Neighbors (KNN)** classification.

The objective is to reduce the dimensionality of hyperspectral data while preserving the most important information and evaluate the resulting classification performance.

## Dataset

The Pavia University scene was acquired by the **ROSIS (Reflective Optics System Imaging Spectrometer)** sensor.

The dataset contains:

* Hyperspectral image: `PaviaU.mat`
* Ground-truth labels: `PaviaU_gt.mat`
* 9 land-cover classes
* Visible and near-infrared spectral information

### Land-Cover Classes

1. Asphalt
2. Meadows
3. Gravel
4. Trees
5. Painted Metal Sheets
6. Bare Soil
7. Bitumen
8. Self-Blocking Bricks
9. Shadows

## Methodology

The processing workflow includes:

1. Loading the hyperspectral image and ground-truth data.
2. Reshaping the hyperspectral cube into a pixel-by-band matrix.
3. Applying **Principal Component Analysis (PCA)**.
4. Reducing the data to the first 10 principal components.
5. Splitting the dataset into training and testing subsets.
6. Training a **KNN classifier**.
7. Evaluating classification accuracy.
8. Producing a classification map.

## PCA Dimensionality Reduction

PCA transforms the original spectral features into a new set of orthogonal principal components.

The first 10 principal components were used for classification.

The explained variance ratios obtained from PCA were:

```text
0.583180635
0.361006948
0.044375610
0.003008413
0.002097923
0.001777430
0.001231210
0.000681950
0.000463220
0.000323406
```

These components preserve the majority of the variance contained in the original spectral data.

## Classification

The reduced feature space was classified using:

```text
Dimensionality Reduction: PCA
Number of Components: 10
Classifier: K-Nearest Neighbors (KNN)
Number of neighbors: 5
Train/Test split: 80/20
Random state: 42
```

## Results

The achieved classification accuracy was:

| Method    |   Accuracy |
| --------- | ---------: |
| PCA + KNN | **82.77%** |

The PCA-based approach achieved higher classification accuracy than the ANOVA F-value approach used in the corresponding experiment.

## Technologies

* Python
* NumPy
* SciPy
* Scikit-learn
* Matplotlib
* Seaborn

## Project Structure

```text
pavia-university-pca-knn/
│
├── README.md
├── pca_knn.py
│
├── data/
│   └── README.md
│
└── results/
    └── classification_map.png
```

> The original hyperspectral dataset is not included in this repository.

## How to Run

Install the required dependencies:

```bash
pip install numpy scipy matplotlib seaborn scikit-learn
```

Place the following files in the appropriate data directory:

```text
PaviaU.mat
PaviaU_gt.mat
```

Then run:

```bash
python pca_knn.py
```

## Comparison

The two feature-processing approaches produced the following results:

| Approach            |   Accuracy |
| ------------------- | ---------: |
| ANOVA F-value + KNN |     78.04% |
| PCA + KNN           | **82.77%** |

In this experiment, PCA provided better classification performance than ANOVA F-value feature selection.

## Conclusion

The results demonstrate the importance of dimensionality reduction and feature selection in hyperspectral image classification.

PCA achieved a higher accuracy by transforming the original spectral information into a reduced set of principal components while retaining most of the variance.

---

**Keywords:** Hyperspectral Remote Sensing · Pavia University · ROSIS · PCA · Dimensionality Reduction · KNN · Image Classification · Remote Sensing · Machine Learning
