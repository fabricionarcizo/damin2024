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

"""This script demonstrates how to create the confusion matrix for multiclass
classification using Python libraries."""

# Import the required libraries.
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# Step 1: Generate a dataset for multiclass classification (adjusting
# parameters).
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=3,
    n_redundant=1, n_classes=3, random_state=42
)

# Step 2: Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Train a Decision Tree Classifier for multiclass classification.
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Step 4: Make predictions on the test set.
y_pred = classifier.predict(X_test)

# Step 5: Compute the confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Step 6: Display the confusion matrix.
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=classifier.classes_
)
cmap = LinearSegmentedColormap.from_list(
    "custom_yellow", ["#FFFFFF", "#F59B11"])
disp.plot(cmap=cmap)
plt.title("Confusion Matrix for Multiclass Classification")
plt.show()
