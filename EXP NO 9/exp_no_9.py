
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

# Load Iris dataset
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Display dataset
print('Sepal-length, Sepal-width, Petal-length, Petal-width')
print(X)

print('\nClass: 0-Iris-Setosa, 1-Iris-Versicolor, 2-Iris-Virginica')
print(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Train the KNN classifier with K=5
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Make predictions on test data
y_pred = classifier.predict(X_test)

# Display confusion matrix
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Display classification report
print('\nAccuracy Metrics:')
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))
