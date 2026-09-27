import pandas as pd
import numpy as np

# Take number of records from user
n = int(input("Enter number of records: "))

age = []
salary = []
department = []
experience = []

# User input
for i in range(n):
    print("\nEnter details for employee", i + 1)

    age.append(int(input("Age: ")))
    salary.append(float(input("Salary: ")))
    department.append(input("Department: "))
    experience.append(float(input("Years of Experience: ")))

# Create DataFrame
df = pd.DataFrame({
    "Age": age,
    "Salary": salary,
    "Department": department,
    "Years of Experience": experience
})

print("\nOriginal Dataset:")
print(df)

# Deliberately inject missing values
if n >= 4:
    df.loc[1, "Age"] = np.nan
    df.loc[2, "Salary"] = np.nan
    df.loc[3, "Department"] = np.nan
    df.loc[0, "Years of Experience"] = np.nan

print("\nDataset with Missing Values:")
print(df)

# Display missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Preprocessing numerical columns using mean
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["Years of Experience"] = df["Years of Experience"].fillna(
    df["Years of Experience"].mean()
)

# Preprocessing categorical column using mode
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])

print("\nPreprocessed Dataset:")
print(df)

# Check missing values after preprocessing
print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())