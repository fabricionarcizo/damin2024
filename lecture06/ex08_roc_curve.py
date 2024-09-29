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

"""This script demonstrates how to compute the ROC curve and AUC for multiple
classification models using Python libraries."""

# Import the required libraries.
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


# Step 1: Generate a dataset for binary classification.
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_classes=2, random_state=42
)

# Step 2: Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Initialize four classifiers.
decision_tree = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier()
logistic_model = LogisticRegression()
naive_bayes = GaussianNB()

# Step 4: Train the models.
decision_tree.fit(X_train, y_train)
knn.fit(X_train, y_train)
logistic_model.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# Step 5: Predict probabilities for ROC (only for positive class).
y_score_tree = decision_tree.predict_proba(X_test)[:, 1]
y_score_knn = knn.predict_proba(X_test)[:, 1]
y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
y_score_naive_bayes = naive_bayes.predict_proba(X_test)[:, 1]

# Step 6: Compute ROC curves and AUC for each model.
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_score_tree)
roc_auc_tree = auc(fpr_tree, tpr_tree)

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_score_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_score_logistic)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

fpr_naive_bayes, tpr_naive_bayes, _ = roc_curve(y_test, y_score_naive_bayes)
roc_auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes)

# Step 7: Plot ROC curves for the four models.
plt.figure(figsize=(10, 8))
plt.plot(
    fpr_tree, tpr_tree, color="#2580B7",
    label=f"Decision Tree (AUC = {roc_auc_tree:.2f})"
)
plt.plot(
    fpr_knn, tpr_knn, color="#179E86",
    label=f"k-NN (AUC = {roc_auc_knn:.2f})"
)
plt.plot(
    fpr_logistic, tpr_logistic, color="#9EBE5B",
    label=f"Logistic Regression (AUC = {roc_auc_logistic:.2f})"
)
plt.plot(
    fpr_naive_bayes, tpr_naive_bayes, color="#F59B11",
    label=f"Naive Bayes (AUC = {roc_auc_naive_bayes:.2f})"
)

# Step 8: Plot the diagonal line representing random guessing.
plt.plot(
    [0, 1], [0, 1], color="#44546A", linestyle="--", label="Random Guessing")

# Step 9: Finalize the plot.
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate (Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve Comparison of Four Classification Models")
plt.legend(loc="lower right")
plt.grid()
plt.show()
