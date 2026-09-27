import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# User input
n = int(input("Enter number of records: "))

data = []

for i in range(n):
    print("\nEnter details for record", i + 1)
    age = float(input("Enter Age: "))
    salary = float(input("Enter Salary: "))
    data.append([age, salary])

data = np.array(data)

print("\nOriginal Data:")
print(data)

# StandardScaler
standard_scaler = StandardScaler()
standard_data = standard_scaler.fit_transform(data)

print("\nAfter StandardScaler:")
print(standard_data)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_data = minmax_scaler.fit_transform(data)

print("\nAfter MinMaxScaler:")
print(minmax_data)

# Observation
print("\nObservation:")
print("StandardScaler: Mean is approximately 0 and standard deviation is 1.")
print("MinMaxScaler: Values are scaled between 0 and 1.")