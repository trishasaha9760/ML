import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Create the same real-estate dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "Price": [50, 60, 75, 90, 100, 115, 130, 145, 160, 190]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 1. Standard Linear Regression
# -------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)

# -------------------------------
# 2. Polynomial Regression
# -------------------------------
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

poly_pred = poly_model.predict(X_test_poly)

poly_r2 = r2_score(y_test, poly_pred)

# -------------------------------
# Display Results
# -------------------------------
print("Actual Prices:")
print(y_test.values)

print("\nLinear Regression Predictions:")
print(linear_pred)

print("\nPolynomial Regression Predictions:")
print(poly_pred)

print("\nR2 Score - Linear Regression:", linear_r2)
print("R2 Score - Polynomial Regression:", poly_r2)

# Comparison
print("\nComparison:")
if poly_r2 > linear_r2:
    print("Polynomial Regression has a higher R2 score.")
elif poly_r2 < linear_r2:
    print("Linear Regression has a higher R2 score.")
else:
    print("Both models have the same R2 score.")