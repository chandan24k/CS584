import os
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from ..models.model_selector import ModelSelector 

# Load the dataset from CSV
data = pd.read_csv('small_test.csv')  # replace with actual CSV path
X = data[['x0', 'x1', 'x2']].values
y = data['y'].values

# Initialize the model and ModelSelector
model = LinearRegression()
selector = ModelSelector(model, k=5, n_bootstraps=100)

# Evaluate the model using k-fold, bootstrap, and AIC
results = selector.evaluate(X, y)

# Print the results
print("k-fold CV MSE:", results["k_fold_cv"])
print("Bootstrap MSE:", results["bootstrap"])
print("AIC:", results["aic"])
