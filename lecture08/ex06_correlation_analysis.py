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

"""This script demonstrates how to perform correlation analysis using the
Pearson, Spearman, and Kendall's Tau correlation coefficients."""

# Import the required libraries.
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from scipy.stats import pearsonr, spearmanr, kendalltau


# Load real-world dataset: Tips dataset.
data = sns.load_dataset("tips")

# Variables:
# - Pearson: "total_bill" and "tip" (continuous, linear relationship)
# - Spearman: "total_bill" and "size" (non-linear, but monotonic relationship)
# - Kendall's Tau: "day" (ordinal) and "time" (ordinal, with ties expected)

# Display first few rows of the dataset.
print(data.head())

# Pearson correlation (continuous, linear relationship between total_bill and
# tip).
pearson_corr, _ = pearsonr(data["total_bill"], data["tip"])
print(f"\nPearson correlation (total_bill vs tip): {pearson_corr:.3f}.")

# Spearman correlation (non-linear but monotonic relationship between
# total_bill and size).
spearman_corr, _ = spearmanr(data["total_bill"], data["size"])
print(f"Spearman correlation (total_bill vs size): {spearman_corr:.3f}.")

# Kendall's Tau (rank correlation between day and time, both ordinal variables)
# Convert day into numeric ordinal values using LabelEncoder for Kendall's Tau.
label_encoder = LabelEncoder()
data["day_num"] = label_encoder.fit_transform(data["day"])

label_encoder = LabelEncoder()
data["time_num"] = label_encoder.fit_transform(data["time"])

kendall_corr, _ = kendalltau(data["day_num"], data["time_num"])
print(f"Kendall's Tau correlation (day vs time): {kendall_corr:.3f}.")

# Plot for Pearson correlation.
g = sns.PairGrid(data[["total_bill", "tip"]])
g.map_diag(sns.histplot)
g.map_lower(sns.regplot, line_kws={"color": "red"})
g.map_upper(sns.kdeplot)
g.figure.subplots_adjust(top=0.9)
g.figure.suptitle(
    f"Total bill vs. tip in US dollars (Pearson: {pearson_corr:.3f})")
g.add_legend()

# Plot for Spearman correlation.
g = sns.PairGrid(data[["total_bill", "size"]])
g.map_diag(sns.histplot)
g.map_lower(sns.scatterplot)
g.map_upper(sns.stripplot)
g.figure.subplots_adjust(top=0.9)
g.figure.suptitle(
    f"Total bill vs. size of the party (Spearman: {spearman_corr:.3f})")
g.add_legend()

# Plot for Kendall's Tau correlation.
g = sns.PairGrid(data[["day_num", "time_num"]])
g.map_diag(sns.histplot)
g.map_lower(sns.scatterplot)
g.map_upper(sns.swarmplot)
g.figure.subplots_adjust(top=0.9)
g.figure.suptitle(
    f"Day of the week vs. time (Kendall's Tau: {kendall_corr:.3f})")

# Set custom tick labels.
day_labels = data["day"].unique()
time_labels = data["time"].unique()

g.axes[0, 0].set_xticks(range(len(day_labels)))
g.axes[0, 0].set_xticklabels(day_labels)
g.axes[0, 0].set_yticks(range(len(day_labels)))
g.axes[0, 0].set_yticklabels(day_labels, rotation=45)
g.axes[1, 0].set_yticks(range(len(time_labels)))
g.axes[1, 0].set_yticklabels([label[:3] for label in time_labels], rotation=45)
g.axes[1, 1].set_xticks(range(len(time_labels)))
g.axes[1, 1].set_xticklabels(time_labels)

g.add_legend()

plt.show()
