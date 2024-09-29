#
# MIT License
#
# Copyright (c) 2024 Fabricio Batista Narcizo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#/

"""This script demonstrates how to use the Naive Bayes algorithm to classify
data points using the scikit-learn library."""

# Import the required libraries.
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Step 1: Generate a sample dataset with 2 features for binary classification.
X, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2,
    n_redundant=0, n_repeated=0, n_classes=2, random_state=42
)

# Step 2: Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Train a Naive Bayes classifier.
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Step 4: Make predictions on the test set.
y_pred = nb_classifier.predict(X_test)

# Step 5: Evaluate the model.
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Step 6: Plot the scatter plot showing correct and incorrect classifications.
plt.figure(figsize=(10, 6))

# Correctly classified points.
correct_classifications = y_test == y_pred

plt.scatter(
    X_test[correct_classifications & (y_test == 0), 0],
    X_test[correct_classifications & (y_test == 0), 1],
    color="#179E86", marker="o",
    label="Correctly Classified Class 0", alpha=0.6
)

plt.scatter(
    X_test[correct_classifications & (y_test == 1), 0],
    X_test[correct_classifications & (y_test == 1), 1],
    color="#F59B11", marker="s",
    label="Correctly Classified Class 1", alpha=0.6
)

# Incorrectly classified points.
incorrect_classifications = y_test != y_pred

plt.scatter(
    X_test[incorrect_classifications & (y_test == 0), 0],
    X_test[incorrect_classifications & (y_test == 0), 1],
    color="#C03B26", marker="o",
    label="Incorrectly Classified Class 0", alpha=0.6
)

plt.scatter(
    X_test[incorrect_classifications & (y_test == 1), 0],
    X_test[incorrect_classifications & (y_test == 1), 1],
    color="#C03B26", marker="s",
    label="Incorrectly Classified Class 1", alpha=0.6
)

plt.title(f"Naive Bayes Classifier (Accuracy: {accuracy:.2f})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# Display the classification report.
print(f"Accuracy of Naive Bayes Classifier: {accuracy:.2f}")
print("\nClassification Report:")
print(report)
