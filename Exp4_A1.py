import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load real-estate dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Display dataset
print("Dataset:")
print(df.head())

# Use only house area (RM is average number of rooms, used here as the input feature)
X = df[["rm"]]
y = df["medv"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

# Display predictions
print("\nActual Prices:")
print(y_test.head().values)

print("\nPredicted Prices:")
print(y_pred[:5])

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Predict price for a new house
area = float(input("\nEnter house area/value: "))
prediction = model.predict([[area]])

print("Predicted House Price:", prediction[0])

# Plot graph
plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, label="Regression Line")
plt.xlabel("House Area / Feature")
plt.ylabel("House Price")
plt.title("House Price Prediction")
plt.legend()
plt.show()