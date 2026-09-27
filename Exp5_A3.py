import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# Custom Student Dataset
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
    X, y, test_size=0.3, random_state=42
)

# Create and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Probability of Pass
y_probability = model.predict_proba(X_test)[:, 1]

# Calculate ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_probability)

# Calculate AUC
auc_score = roc_auc_score(y_test, y_probability)

print("\nAUC Score:", round(auc_score, 2))

# Plot ROC Curve
plt.plot(fpr, tpr, label="Logistic Regression")

# Random classifier line
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Student Pass/Fail Prediction")
plt.legend()
plt.grid()
plt.show()