import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Custom student dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'Result': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Features and target
X = df[['Study_Hours', 'Attendance']]
y = df['Result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPredicted Results:", y_pred)
print("Actual Results:", y_test.values)
print("Accuracy:", accuracy)

# User input
study = float(input("\nEnter study hours: "))
attendance = float(input("Enter attendance percentage: "))

prediction = model.predict([[study, attendance]])

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")