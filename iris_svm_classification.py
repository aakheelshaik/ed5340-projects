# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:33:40 2023

@author: skaas
"""
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets


# Load the iris dataset
iris = load_iris()

# Extract the features (sepal length and width) and target (flower type)
X = iris.data[:, :2]
y = iris.target
# Plot the scatter plot of sepal length vs sepal width
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='r', label='Setosa')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='g', label='Versicolor')
plt.scatter(X[y == 2, 0], X[y == 2, 1], c='b', label='Virginica')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()
# Split the data into training and testing sets with 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM model
svm = SVC(kernel='linear', C=1, random_state=42)

# Train the SVM model on the training data
svm.fit(X_train, y_train)


# Predict the flower types on the testing data

y_pred = svm.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
import numpy as np

# SVM class definition
class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias with zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient descent optimization
        for epoch in range(self.num_epochs):
            # Compute hinge loss and gradients
            loss = 0
            for i in range(X.shape[0]):
                margin = y[i] * (np.dot(self.weights, X[i]) + self.bias)
                if margin < 1:
                    loss += 1 - margin
                    gradient_w = -y[i] * X[i]
                    gradient_b = -y[i]
                else:
                    gradient_w = 0
                    gradient_b = 0
                self.weights -= self.learning_rate * (gradient_w + 2 * self.lambda_param * self.weights)
                self.bias -= self.learning_rate * gradient_b

            # Print loss for every epoch
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        # Predict class labels for input data X
        return np.sign(np.dot(X, self.weights) + self.bias)

# Function to calculate Precision, Recall, F-1 Score, and Accuracy
def evaluate(y_true, y_pred):
    # Calculate true positive, false positive, true negative, false negative
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    # Calculate Precision, Recall, F-1 Score, and Accuracy
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    return precision, recall, f1_score, accuracy

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    # Calculate confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    # Plot confusion matrix
    print("Confusion Matrix:")
    print(f"True Positive: {tp}")
    print(f"False Positive: {fp}")
    print(f"True Negative: {tn}")
    print(f"False Negative: {fn}")


# Load the digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Visualize the data
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})
for ax, image, label in zip(axes.flat, digits.images, digits.target):
    ax.imshow(image, cmap='binary')
    ax.set_title(f'Target: {label}')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Test the model
y_pred = svm.predict(X_test)

# Calculate accuracy, precision, recall, and F-1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1_score = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

# Plot confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
plt.matshow(confusion_mat, cmap=plt.cm.gray)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
