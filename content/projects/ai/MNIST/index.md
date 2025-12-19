+++
title = "Dimensionality Reduction of the MNIST Dataset"
summary = "Exploring Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA) and UMAP to perform dimensionality reduction on the MNIST dataset."
description = ""
featuredImage = ""
tags = ["Dimensionality Reduction", "MNIST", "PCA", "LDA", "UMAP"]
categories = ["AI"]
collections = [""]
weight = 8
draft = false
+++

Dimensionality Reduction is a powerful unsupervised machine learning technique that is widely used in data analytics and data science to help visualize data, select good features, and to train models efficiently. We use dimensionality reduction to take higher-dimensional data and represent it in a lower dimension. There are many dimensionality reduction algorithms to choose from and no single best algorithm for all cases. 

In this project, I explored three dimensionality reduction algorithms and different configurations for each algorithm.​​​​​​​

{{< button href="https://colab.research.google.com/drive/1Z1JWLEr5QgmlogpN_zGx5FxZU0w3Ylvn" target="_self" >}}
{{< icon "link" >}} View on Google Colab
{{< /button >}}

## Load and Preprocess the Dataset

After loading the dataset, I noticed that there are 42,000 rows and 785 columns, among which 784 features were contributed by 28 x 28 pixels from each MNIST image and one label column. The label tells us the class (0 - 9) of each image. To compare the classification accuracy later, I only used the training dataset that has the label.

First, I separated the features and the label, and applied the ***StandardScaler*** to standardize the features. I also used ***matplotlib*** to display some of the numbers to get a better understanding of the data.

![](01.Preprocess.jpeg)

## Dimensionality Reduction Techniques

The methods I chose were PCA, LDA and UMAP. After implementing each method, I used ***matplotlib*** to create a scatterplot to represent the first two components and color-coded labels.

### Principal Component Analysis (PCA)

PCA is an unsupervised, linear transformation algorithm that projects the original features onto a smaller set of features while retaining most of the information.

I chose this method because it can preserve the global structure, and it's very fast and memory-efficient. I implemented it using ***sklearn.decomposition.PCA***.

![](02.PCA.jpeg)

### Linear Discriminant Analysis (LDA)

LDA is also a linear transformation method like PCA, but the difference is that LDA is a supervised method that uses labels to maximize the separation between classes in a lower dimensional space. I implemented it using ***sklearn.discriminant_analysis.LinearDiscriminantAnalysis***.

![](03.LDA.jpeg)

### UMAP

UMAP is a nonlinear, manifold-based method that seeks to preserve local data structure.​​​​​​​

![](04.UMAP.jpeg)

## Evaluation

I used several metrics to evaluate these three methods, first I used **K-Means Clustering** on each reduced dataset to visualize the clusters of different digits and to evaluate their quality.

I used two numerical metrics to show how well each method performed:
- **Silhouette Score**: Measures how similar points are to their own cluster vs other clusters (higher is better).
- **Davies-Bouldin Score**: Measures the average similarity between clusters (lower is better).

To measure how well the reduced data preserves class information, I trained a **K-Nearest Neighbors (KNN)** classifier and tested it on each reduced dataset to give an accuracy score for each method.​​​​​​​

![](05.Evaluation.jpeg)

### Silhouette Score

Ranges from -1 to 1, a higher silhouette scores (close to 1) indicate better defined clusters. Negative scores indicate poor or incorrect clustering.

The result shows that of the three methods, UMAP is the best at preserving the data's structure.
- PCA has low separability between clusters.
- LDA has a slight improvement over PCA.
- UMAP has the highest separability.

### Davies-Bouldin Score

Lower values indicate better separation. The result shows that of the three methods, UMAP also has the best cluster separation.
- PCA has poor separation and compactness.
- LDA has a better separation than PCA.
- UMAP has the best separation and compactness.

### Comparing Classification Accuracy

Classification accuracy can show how well the reduced data preserves the original information. Higher accuracy indicates better preservation of class information in the reduced space.

The high accuracy score of UMAP indicates that it has effectively preserved class separability, which aligns with its reputation for capturing complex structures in high-dimensional data.
- PCA has low class structure preservation.
- LDA has moderate class structure preservation.
- UMAP has excellent class structure preservation.

## Summary

For MNIST dimensionality reduction, UMAP significantly outperforms both PCA and LDA. As a non-linear, unsupervised method, it achieves a high silhouette score and the lowest Davies-Bouldin score. Classification accuracy also shows that UMAP preserves class information effectively.