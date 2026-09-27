import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# User-defined dataset
data = {
    'Age': [22, 25, None, 30, 28],
    'Salary': [25000, 35000, 45000, None, 50000],
    'YearsExperience': [1, 2, 4, 5, None]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Numerical columns
num_columns = ['Age', 'Salary', 'YearsExperience']

# Pipeline: Missing value handling + MinMaxScaler
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Apply pipeline
transformed_data = pipeline.fit_transform(df[num_columns])

print("\nTransformed Data using MinMaxScaler:")
print(transformed_data)