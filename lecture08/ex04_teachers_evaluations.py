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

"""This script demonstrates how to use the Linear Regression model to predict
the scores of professors based on multiple features."""

# Import the required libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# Set the random seed for reproducibility.
np.random.seed(42)

# Load the dataset as a DataFrame.
df = pd.read_csv("./lecture08/data/evals.csv")
columns = ["age", "gender", "bty_avg", "pic_outfit", "pic_color", "score"]
colors = ["#179E86", "#2580B7", "#9EBE5B", "#C03B26", "#44546A"]

# Drop rows with missing values in the columns of interest.
df = df.dropna(subset=columns)


# Create a scatter plot to show the relationship between bty_avg and score.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=columns[2], y=columns[-1], color=colors[0])
plt.title("Relationship between Beauty Average and Score")
plt.xlabel("Beauty Average")
plt.ylabel("Score")
plt.show()


# Create a scatter plot to show the relationship between bty_avg and score and
# modifying the circle size to show data overlapping.
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x=columns[2], y=columns[-1], color=colors[0],
    size=df.groupby([columns[2], columns[-1]]).transform("size"),
    sizes=(25, 250)
)
plt.title("Relationship between Beauty Average and Score")
plt.xlabel("Beauty Average")
plt.ylabel("Score")
plt.show()


# Create a scatter plot to show the relationship between bty_avg and score with
# jitter applied.
JITTER_STRENGTH = 0.1
x_jitter = df[columns[2]] + np.random.normal(
    0, JITTER_STRENGTH, size=len(df))
y_jitter = df[columns[-1]] + np.random.normal(
    0, JITTER_STRENGTH, size=len(df))

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=x_jitter, y=y_jitter, color=colors[0]
)
plt.title("Relationship between Beauty Average and Score")
plt.xlabel("Beauty Average")
plt.ylabel("Score")
plt.show()


# Apply a simple linear regression.
X = x_jitter.values.reshape(-1, 1)
y = y_jitter.values

# Create and fit the model.
model = LinearRegression()
model.fit(X, y)

# Predict the scores.
y_pred = model.predict(X)

# Plot the data points.
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=x_jitter, y=y_jitter, color=colors[0], label="Data points"
)

# Plot the regression line.
plt.plot(X, y_pred, color=colors[3], label="Regression line")

plt.title("Relationship between Beauty Average and Score with Regression Line")
plt.xlabel("Beauty Average")
plt.ylabel("Score")
plt.legend()
plt.show()

# Print the linear equation.
intercept = model.intercept_
slope = model.coef_[0]
print(f"\nThe linear equation is: y = {slope:.4f}x + {intercept:.4f}.")

# Calculate the Total Sum of Squares (TSS).
tss = ((y - y.mean()) ** 2).sum()
print(f"The Total Sum of Squares (TSS) is: {tss:.4f}.")

# Calculate the Sum of Squared Errors (SSE).
sse = ((y - y_pred) ** 2).sum()
print(f"The Sum of Squared Errors (SSE) is: {sse:.4f}.")

# Calculate the Coefficient of Determination (R^2).
r_squared = (tss - sse) / tss
print(f"The Coefficient of Determination (R^2) is: {r_squared:.4f}.")


# Apply a multiple linear regression with bty_avg and age.
X_multi = df[[columns[2], columns[0]]]
y_multi = df[columns[-1]]

# Create and fit the model.
model = LinearRegression()
model.fit(X_multi, y_multi)

# Predict the scores.
y_pred_multi = model.predict(X_multi)

# Plot the regression plane and residuals.
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X_train = [X_multi[columns[2]].values, X_multi[columns[0]].values]
y_train = y_multi.values

# Scatter plot of the actual data points.
ax.scatter(
    X_train[0], X_train[1], y_train,
    color="#2580B7", label="Training data"
)

# Create a meshgrid for the regression plane.
x_surf, y_surf = np.meshgrid(
    np.linspace(X_train[0].min(), X_train[0].max(), 10),
    np.linspace(X_train[1].min(), X_train[1].max(), 10)
)
z_surf = model.intercept_ + model.coef_[0] * x_surf + model.coef_[1] * y_surf

# Plot the regression plane.
ax.plot_surface(
    x_surf, y_surf, z_surf,
    color="#C03B26", alpha=0.5, label="Regression plane"
)

ax.set_xlabel(columns[2])
ax.set_ylabel(columns[0])
ax.set_zlabel(columns[-1])
plt.title(
    "Multiple Linear Regression on Professor Evaluations and Beauty Dataset")
plt.legend()
plt.show()


# Print the linear equation.
intercept = model.intercept_
slope = model.coef_
print("\nThe linear equation is: y = "
      f"{slope[0]:.4f}x1 + {slope[1]:.4f}x2 + {intercept:.4f}."
)

# Calculate the Total Sum of Squares (TSS).
tss_multi = ((y_multi - y_multi.mean()) ** 2).sum()
print(f"The Total Sum of Squares (TSS) is: {tss_multi:.4f}.")

# Calculate the Sum of Squared Errors (SSE).
sse_multi = ((y_multi - y_pred_multi) ** 2).sum()
print(f"The Sum of Squared Errors (SSE) is: {sse_multi:.4f}.")

# Calculate the Coefficient of Determination (R^2).
r_squared_multi = (tss_multi - sse_multi) / tss_multi
print(f"The Coefficient of Determination (R^2) is: {r_squared_multi:.4f}.")


# Apply a multiple linear regression with bty_avg and gender.
X_multi = df[[columns[2], columns[1]]]
y_multi = df[columns[-1]]

label_encoder = LabelEncoder()
X_multi = X_multi.copy()
X_multi[columns[1]] = label_encoder.fit_transform(X_multi[columns[1]])

# Create and fit the model.
model = LinearRegression()
model.fit(X_multi, y_multi)

# Predict the scores.
y_pred_multi = model.predict(X_multi)

# Plot the data points with hue based on gender.
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_multi[columns[2]], y=y_multi,
    hue=X_multi[columns[1]].map({0: "Female", 1: "Male"}),
    palette=colors[1:3], legend="full"
)

# Plot the regression lines for each gender.
genders = ["Female", "Male"]
for gender in X_multi[columns[1]].unique():
    mask = X_multi[columns[1]] == gender
    X_gender = X_multi[mask]
    y_pred_gender = model.predict(X_gender)
    plt.plot(
        X_gender[columns[2]], y_pred_gender, color=colors[1:3][gender],
        label=f"Regression line (Gender {genders[gender]})"
    )

plt.legend()
plt.title("Relationship between Beauty Average/Gender and Score")
plt.xlabel("Beauty Average")
plt.ylabel("Score")
plt.show()


# Print the linear equation.
intercept = model.intercept_
slope = model.coef_
print("\nThe linear equation is: y = "
      f"{slope[0]:.4f}x1 + {slope[1]:.4f}x2 + {intercept:.4f}."
)

# Calculate the Total Sum of Squares (TSS).
tss_multi = ((y_multi - y_multi.mean()) ** 2).sum()
print(f"The Total Sum of Squares (TSS) is: {tss_multi:.4f}.")

# Calculate the Sum of Squared Errors (SSE).
sse_multi = ((y_multi - y_pred_multi) ** 2).sum()
print(f"The Sum of Squared Errors (SSE) is: {sse_multi:.4f}.")

# Calculate the Coefficient of Determination (R^2).
r_squared_multi = (tss_multi - sse_multi) / tss_multi
print(f"The Coefficient of Determination (R^2) is: {r_squared_multi:.4f}.")


# Apply a multiple linear regression with age, gender, bty_avg, pic_outfit, and pic_color.
X_multi = df[columns[:-1]]
y_multi = df[columns[-1]]

# Encode categorical variables.
label_encoder = LabelEncoder()
X_multi = X_multi.copy()
X_multi["gender"] = label_encoder.fit_transform(X_multi["gender"])
X_multi["pic_outfit"] = label_encoder.fit_transform(X_multi["pic_outfit"])
X_multi["pic_color"] = label_encoder.fit_transform(X_multi["pic_color"])

# Create and fit the model.
model = LinearRegression()
model.fit(X_multi, y_multi)

# Predict the scores.
y_pred_multi = model.predict(X_multi)

# Print the linear equation.
intercept = model.intercept_
slope = model.coef_

# Add a constant to the model (intercept).
X_multi = sm.add_constant(X_multi)

# Fit the model using statsmodels.
model_sm = sm.OLS(y_multi, X_multi).fit()

# Get the confidence intervals for each coefficient.
conf_intervals = model_sm.conf_int()
print("\nConfidence intervals for each coefficient:")
print(conf_intervals)

# Plot the coefficients as dots and the confidence interval as a range.
plt.figure(figsize=(10, 6))
coefficients = model_sm.params[1:]  # Exclude the intercept
conf_intervals = conf_intervals[1:]  # Exclude the intercept

# Sort coefficients and confidence intervals by the smallest coefficient values
sorted_indices = np.argsort(coefficients)
sorted_coefficients = coefficients[sorted_indices]
sorted_conf_intervals = conf_intervals.iloc[sorted_indices]

# Plot the coefficients.
plt.errorbar(
    y=range(len(sorted_coefficients)), x=sorted_coefficients,
    xerr=[ sorted_coefficients - sorted_conf_intervals[0],
           sorted_conf_intervals[1] - sorted_coefficients ],
    fmt="o", color=colors[0], ecolor=colors[-1], elinewidth=2, capsize=4
)
plt.yticks(
    range(len(sorted_coefficients)),
    X_multi.columns[1:][sorted_indices], rotation=45
)

plt.yticks(
    range(len(coefficients)), X_multi.columns[1:][sorted_indices], rotation=45
)
plt.title("Coefficients and Confidence Intervals")
plt.ylabel("Features")
plt.xlabel("Coefficient Value")
plt.show()

# Print the linear equation.
print("\nThe linear equation is: y = "
    f"{slope[0]:.4f}*age + {slope[1]:.4f}*gender + {slope[2]:.4f}*bty_avg + "
    f"{slope[3]:.4f}*pic_outfit + {slope[4]:.4f}*pic_color + {intercept:.4f}."
)

# Calculate the Total Sum of Squares (TSS).
tss_multi = ((y_multi - y_multi.mean()) ** 2).sum()
print(f"The Total Sum of Squares (TSS) is: {tss_multi:.4f}.")

# Calculate the Sum of Squared Errors (SSE).
sse_multi = ((y_multi - y_pred_multi) ** 2).sum()
print(f"The Sum of Squared Errors (SSE) is: {sse_multi:.4f}.")

# Calculate the Coefficient of Determination (R^2).
r_squared_multi = (tss_multi - sse_multi) / tss_multi
print(f"The Coefficient of Determination (R^2) is: {r_squared_multi:.4f}.")
