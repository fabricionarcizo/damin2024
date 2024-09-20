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

"""This script demonstrates how to create basic visualizations using Matplotlib
and Seaborn."""

# Import the required libraries.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Read the sample_data.csv file.
df = pd.read_csv("./lecture05/data/sample_data_100.csv")

# Define the plot style.
with plt.style.context("fast"):

    # Set up the layout for plots.
    fig = plt.figure(constrained_layout=True, figsize=(20, 6))
    gs = fig.add_gridspec(2, 4)

    # Bar Chart: Distribution of "Category".
    ax1 = fig.add_subplot(gs[0, 0])
    df["Category"].value_counts().sort_index().plot(
        kind="bar", ax=ax1, color="#2580B7")
    ax1.set_title("Bar Chart: Category Distribution")
    ax1.set_ylabel("Count")

    # Line Graph: Plot Date vs Value2.
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df["Date"], df["Value2"], color="#179E86")
    ax2.set_xticks(df["Date"][::len(df)//3])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    ax2.set_title("Line Graph: Date vs Value2")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value2")

    # Pie Chart: Distribution of "Flag".
    ax3 = fig.add_subplot(gs[0, 2])
    df["Flag"].value_counts().plot(
        kind="pie", ax=ax3, autopct="%1.1f%%", colors=["#9EBE5B", "#F59B11"])
    ax3.set_title("Pie Chart: Flag Distribution")
    ax3.set_ylabel("")

    # Scatter Plot: Value1 vs Value3.
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(df["Value1"], df["Value3"], color="#C03B26")
    ax4.set_title("Scatter Plot: Value1 vs Value3")
    ax4.set_xlabel("Value1")
    ax4.set_ylabel("Value3")

    # Histogram: Distribution of Value4.
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(df["Value4"], bins=15, color="#633248", edgecolor="white")
    ax5.set_title("Histogram: Value4")
    ax5.set_xlabel("Value4")
    ax5.set_ylabel("Frequency")

    # Heatmap: Correlation between numerical variables
    ax5 = fig.add_subplot(gs[1, 1:3])
    sns.heatmap(df.loc[:, 'Value1':'Score2'].drop("Date", axis=1).corr(),
                annot=True, fmt=".2f", cmap="vlag", ax=ax5)
    ax5.set_title("Heatmap: Correlation between Variables")

    # Box Plot: Distribution of Value5 by Category.
    ax6 = fig.add_subplot(gs[1, 3])
    sns.boxplot(x="Category", y="Value5", data=df, ax=ax6, palette="Set2")
    ax6.set_title("Box Plot: Value5 by Category")

    # Display the plots
    plt.show()
