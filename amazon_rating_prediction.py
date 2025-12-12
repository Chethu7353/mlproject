# My First ML Project using real dataset from Kaggle
# Title: "Amazon Product Rating Prediction Using Linear Regression"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("amazon_products.csv")     # Use relative path

# ---------------- CLEAN PRICE ----------------
df["product_price"] = (
    df["product_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
)

# ---------------- CLEAN RATING ----------------
df["product_star_rating"] = (
    df["product_star_rating"]
    .astype(str)
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
)

# ---------------- DROP MISSING VALUES ----------------
df = df.dropna(subset=["product_price", "product_star_rating"])

# ---------------- FEATURES ----------------
x = df["product_price"]
y = df["product_star_rating"]

# Convert x into 2D array for sklearn
X = x.values.reshape(-1, 1)

# ---------------- LINEAR REGRESSION ----------------
reg = LinearRegression()
reg.fit(X, y)

# Predict ratings
y_pred = reg.predict(X)

# ---------------- PLOTS ----------------

def normal_scatter_graph():
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color="red")
    plt.title("Amazon Rating vs Price")
    plt.xlabel("Price")
    plt.ylabel("Rating")
    plt.show()

def regression_graph():
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color="red", label="Actual Data")

    # Sort values for smooth regression line
    sorted_index = np.argsort(x)
    plt.plot(x.iloc[sorted_index], y_pred[sorted_index], color="blue", label="Regression Line")

    plt.title("Linear Regression: Rating Prediction from Price")
    plt.xlabel("Price")
    plt.ylabel("Rating")
    plt.legend()
    plt.show()

# ---------------- MAIN FUNCTION ----------------

def main():
    print("\n--- Amazon Rating Prediction Model ---\n")
    print("Slope (Coefficient):", reg.coef_[0])
    print("Intercept:", reg.intercept_)
    print("R² Score:", r2_score(y, y_pred))

    # User input handling
    try:
        k = float(input("\nEnter price to predict rating: "))
        predicted_value = reg.predict([[k]])[0]
        print("Predicted Rating:", round(predicted_value, 2))
    except ValueError:
        print("Invalid price entered!")

    # Graph options
    show_scatter = input("\nShow scatter graph? (yes/no): ").strip().lower()
    show_regression = input("Show regression graph? (yes/no): ").strip().lower()

    if show_scatter == "yes":
        normal_scatter_graph()

    if show_regression == "yes":
        regression_graph()

if __name__ == "__main__":
    main()
