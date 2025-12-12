Amazon Product Rating Prediction
A Simple ML Project Using Linear Regression (Kaggle Dataset)

This project predicts Amazon product star ratings based on product price using Linear Regression.
It demonstrates data cleaning, visualization, model training, and prediction — perfect as a first ML project.

🚀 Features

Cleans messy price values (₹ symbol, commas, text)

Extracts numeric star ratings

Removes missing data

Visualizes:

Scatter plot (Price vs Rating)

Regression line plot

Trains a Linear Regression model using scikit-learn

Predicts product ratings based on user-entered price

Displays model evaluation (R² score)

📁 Dataset

Dataset used: amazon_products.csv
Source: Kaggle
Make sure the file is placed in the project folder.

🧰 Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-learn

📊 Visualizations
1️⃣ Scatter Plot

Shows how ratings vary with price.

2️⃣ Scatter Plot + Regression Line

Displays the fitted linear regression relationship.

🔍 Model Details

Coefficient (Slope): Shows how rating changes with price

Intercept: Base rating

R² Score: Indicates how well the model fits the data

▶️ How to Run the Project
1. Install the required libraries
pip install -r requirements.txt

2. Run the script
python amazon_rating_prediction.py

3. Enter a price when prompted

Example:

Enter price to predict rating: 999
Predicted Rating: 4.02

4. Choose to view graphs

Scatter graph

Regression graph

📂 Project Structure
ML_Project_Amazon/
│── amazon_products.csv
│── amazon_rating_prediction.py
│── requirements.txt
│── README.md

✨ Future Improvements

Add more features (brand, category, review count)

Build a web app using Streamlit or Flask

Use advanced models (Random Forest, XGBoost)

Add data preprocessing + normalization
