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

"""This script demonstrates how to create advanced visualizations using Python
libraries."""

# Import the required libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

from wordcloud import WordCloud


# Load the CSV file.
df = pd.read_csv("./lecture05/data/sample_data_1000.csv")

# Define the plot style.
with plt.style.context("fast"):

    # Pair Plot (Scatterplot Matrix).
    pair_plot = sns.pairplot(df.iloc[:, 1:6])
    plt.show()

    # Violin Plot.
    plt.figure(figsize=(8, 8))
    sns.violinplot(x="Category", y="Value4", data=df, palette="Set2",
                   order=sorted(df["Category"].unique()))
    plt.title("Violin Plot: Value4 by Category")
    plt.show()

    # Density Plot.
    plt.figure(figsize=(8, 8))
    sns.kdeplot(df["Value4"], shade=True, color="#179E86")
    plt.title("Density Plot: Value4")
    plt.show()

    # Bubble Plot (scatter plot with bubble size based on "Value6").
    plt.figure(figsize=(8, 8))
    plt.scatter(df["Value1"], df["Value2"], s=df["Value6"], alpha=0.5,
                color="#9EBE5B")
    plt.title("Bubble Plot: Value1 vs Value2 (size by Value6)")
    plt.xlabel("Value1")
    plt.ylabel("Value2")
    plt.show()

    # Treemap.
    plt.figure(figsize=(8, 8))
    treemap_data = df["Category"].value_counts()
    squarify.plot(sizes=treemap_data, label=treemap_data.index, alpha=.8)
    plt.title("Treemap: Category Distribution")
    plt.axis("off")
    plt.show()

    # Radar Chart (Spider Plot).
    categories = ["Score1", "Score2", "Score3"]
    radar_data = df[categories].mean().tolist()

    # Radar chart data.
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    radar_data += radar_data[:1]
    angles += angles[:1]

    # Plot radar chart.
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={ "polar": True })
    ax.fill(angles, radar_data, color="#F59B11", alpha=0.25)
    ax.plot(angles, radar_data, color="#F59B11", linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title("Radar Chart: Average Scores")
    plt.show()

    # Word Cloud.
    plt.figure(figsize=(16, 8))
    TEXT = " ".join(df["Word1"].values)
    wordcloud = WordCloud(
        width=800, height=400, background_color="white").generate(TEXT)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    plt.show()
