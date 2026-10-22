
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Iris dataset directly
iris = load_iris()

names = [
    'Sepal_Length',
    'Sepal_Width',
    'Petal_Length',
    'Petal_Width'
]

X = pd.DataFrame(iris.data, columns=names)
y = iris.target

# Display sample data
dataset = X.copy()
dataset['Class'] = [
    'Iris-' + iris.target_names[i] for i in y
]

print('Sample instances from the dataset:')
print(dataset.head())

print('\nAttributes and datatypes:')
print(dataset.dtypes)

print('\nTotal samples:', len(dataset))

# Plot colors
colormap = np.array(['red', 'lime', 'black'])

plt.figure(figsize=(14, 7))

# Real plot
plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(
    X['Petal_Length'],
    X['Petal_Width'],
    c=colormap[y]
)

# K-Means clustering
model = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans_labels = model.fit_predict(X)

plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(
    X['Petal_Length'],
    X['Petal_Width'],
    c=colormap[kmeans_labels]
)

# Match cluster IDs to actual class IDs for fair evaluation
def calculate_accuracy(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    rows, cols = linear_sum_assignment(-cm)
    return cm[rows, cols].sum() / len(true_labels)

print('\nThe accuracy score of K-Means:',
      calculate_accuracy(y, kmeans_labels))
print('The Confusion matrix of K-Means:\n',
      confusion_matrix(y, kmeans_labels))

# Gaussian Mixture Model (EM)
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_labels = gmm.fit_predict(X)

plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(
    X['Petal_Length'],
    X['Petal_Width'],
    c=colormap[gmm_labels]
)

print('\nThe accuracy score of EM:',
      calculate_accuracy(y, gmm_labels))
print('The Confusion matrix of EM:\n',
      confusion_matrix(y, gmm_labels))

plt.tight_layout()
plt.show()
