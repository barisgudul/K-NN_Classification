# K-Nearest Neighbors Classification Project

This project focuses on utilizing the K-Nearest Neighbors (K-NN) algorithm to accurately classify data, showcasing its simplicity and effectiveness in solving classification problems across various domains.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Visualization](#visualization)
- [Results](#results)
- [K-Nearest Neighbors](#k-nearest-neighbors)

## Overview

This project aims to predict classifications using the K-Nearest Neighbors (K-NN) algorithm. It addresses [specific problem] by providing an accurate and efficient solution through classification techniques, highlighting the effectiveness of K-NN in handling such tasks.

## Features

- Implementation of the K-NN classification algorithm
- Feature scaling and optimization for accuracy
- Visualization of training and testing phases

## Visualization

### Training Phase

![Training Visualization](K-Nearest%20Neighbors/train.png)

The training phase visualization demonstrates how the K-NN algorithm separates data points into distinct classes. Blue points represent one class, while orange points represent another. The decision boundary, shown in the background, illustrates how the algorithm distinguishes between the classes based on the training data.

### Testing Phase

![Testing Visualization](K-Nearest%20Neighbors/test.png)

The testing phase visualization showcases the performance of the trained K-NN model on unseen data. Similar to the training phase, the decision boundary divides the classes, with blue and orange points representing their respective classes. This plot highlights the model's ability to generalize and classify test data accurately.

## Results

The model was tested on a sample input of `[30, 87000]`, representing an individual with an age of 30 and an estimated salary of 87,000. The classifier predicted the class as `[1]`, indicating the individual belongs to the positive class.

### Confusion Matrix and Accuracy Score

The confusion matrix and accuracy score for the model on the test dataset are as follows:

**Confusion Matrix:**
```
[[48  4]
 [ 3 25]]
```
This matrix shows the number of true positives, true negatives, false positives, and false negatives, providing insight into the model's classification performance.

**Accuracy Score:**
```
0.9125
```
This score indicates that the model correctly classified 91.25% of the test samples, demonstrating high accuracy and effectiveness.

### Prediction and Ground Truth Comparison

The following table compares the predicted values (`y_pred`) with the actual values (`y_test`) from the test dataset:

```
[[1 0]
 [1 1]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [0 0]
 [1 0]
 [1 1]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [1 1]
 [1 1]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [0 1]
 [0 0]
 [1 1]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [0 0]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [0 0]
 [0 0]
 [1 1]
 [1 0]
 [0 0]
 [0 0]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [1 1]
 [0 1]
 [1 1]
 [0 0]
 [1 0]
 [1 1]
 [0 0]
 [0 0]
 [0 0]
 [1 1]
 [0 0]
 [1 1]
 [1 1]
 [0 0]
 [0 1]]
```

This comparison highlights the model's accuracy and areas where it may need further tuning or optimization.

## K-Nearest Neighbors

The K-Nearest Neighbors (K-NN) algorithm is implemented in this project using the `KNeighborsClassifier` from the `sklearn.neighbors` library. The classifier is initialized with 5 neighbors (`n_neighbors=5`), the Minkowski metric (`metric="minkowski"`), and a parameter `p=2`, which corresponds to the Euclidean distance. Key aspects include:
- Selection of optimal number of neighbors (`k`) to balance bias and variance
- Use of Minkowski distance with `p=2` for determining similarity
- Results demonstrating the model's performance in terms of accuracy and other metrics

The classifier was trained using the `fit` method with `X_train` as the input features and `y_train` as the target labels.- Selection of optimal number of neighbors (`k`) to balance bias and variance
- Use of distance metrics (e.g., Euclidean, Manhattan) for determining similarity
- Results demonstrating the model's performance in terms of accuracy and other metrics
