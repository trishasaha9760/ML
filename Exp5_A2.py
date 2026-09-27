import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

# Custom student dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'Result': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Study_Hours', 'Attendance']]
y = df['Result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Get probability of Pass (class 1)
probabilities = model.predict_proba(X_test)[:, 1]

# Test different thresholds
thresholds = [0.3, 0.5, 0.7]

print("Threshold\tPrecision\tRecall")
print("------------------------------------")

for threshold in thresholds:

    # Convert probabilities into 0/1 predictions
    y_pred = (probabilities >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"{threshold}\t\t{precision:.2f}\t\t{recall:.2f}")