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

"""This script demonstrates how to evaluate a model using the accuracy,
precision, recall, and F1-score metrics."""

# Import the required libraries.
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)


# Assuming you have your predicted labels and true labels stored in variables.
predicted_labels = [0, 1, 1, 0, 1, 1, 1, 0, 0]
true_labels = [0, 1, 0, 1, 1, 1, 1, 0, 1]

# Calculate accuracy.
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Calculate precision.
precision = precision_score(true_labels, predicted_labels)
print("Precision:", precision)

# Calculate recall.
recall = recall_score(true_labels, predicted_labels)
print("Recall:", recall)

# Calculate F1-score.
f1 = f1_score(true_labels, predicted_labels)
print("F1-score:", f1)
