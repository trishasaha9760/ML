import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create a sample real-estate dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "Price": [50, 60, 75, 90, 100, 115, 130, 145, 160, 190]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Independent variables
X = df[["Area", "Bedrooms"]]

# Dependent variable
y = df["Price"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict prices
y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

# Model coefficients
print("\nIntercept:", model.intercept_)
print("Area Coefficient:", model.coef_[0])
print("Bedrooms Coefficient:", model.coef_[1])

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# User input
area = float(input("\nEnter house area in sq.ft: "))
bedrooms = int(input("Enter number of bedrooms: "))

prediction = model.predict([[area, bedrooms]])

print("Predicted House Price:", prediction[0], "Lakhs")