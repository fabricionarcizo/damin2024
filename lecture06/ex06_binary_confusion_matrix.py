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

"""This script demonstrates how to create the confusion matrix for binary
classification using Python libraries."""

# Import the required libraries.
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Step 1: Generate a dataset for binary classification.
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3,
    n_redundant=1, n_classes=2, random_state=42
)

# Step 2: Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Train a Decision Tree Classifier for binary classification.
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Step 4: Make predictions on the test set.
y_pred = classifier.predict(X_test)

# Step 5: Compute the confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Step 6: Plot the confusion matrix with TP, FN, FP, TN.
plt.figure(figsize=(8, 6))
cmap = LinearSegmentedColormap.from_list(
    "custom_green", ["#FFFFFF", "#179E86"])
ax = sns.heatmap(
    cm, annot=True, fmt="d", cmap=cmap,
    xticklabels=["Predicted Positive", "Predicted Negative"],
    yticklabels=["Actual Positive", "Actual Negative"]
)

# Annotate the confusion matrix with TP, TN, FP, FN.
ax.text(
    0.2, 0.2, "TP", horizontalalignment="center",
    verticalalignment="center", color="white", fontsize=14
)
ax.text(
    0.2, 1.8, "FP", horizontalalignment="center",
    verticalalignment="center", color="black", fontsize=14
)
ax.text(
    1.8, 0.2, "FN", horizontalalignment="center",
    verticalalignment="center", color="black", fontsize=14
)
ax.text(
    1.8, 1.8, "TN", horizontalalignment="center",
    verticalalignment="center", color="white", fontsize=14
)

plt.title("Confusion Matrix with TP, FN, FP, and TN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
