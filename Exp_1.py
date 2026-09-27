import numpy as np
import pandas as pd

# NumPy Array Operations
marks= np.array([72,85,91,68,71])

print("Marks:", marks)
print("Mean:", np.mean(marks))
print("Maximum:", np.max(marks))
print("Minimum:", np.min(marks))

# Pandas DataFrame Creation
data = {
     "Name":["Amit", "Riya" ,"Sourav", "Neha", "Rahul"],
     "Attendance":[88,92,76,95,81],
     "Marks":[72,85,68,91,77]
}

df = pd.DataFrame(data)

# Data Exploration
print("\n---First Five Records---")
print(df.head())

print("\n---Data Information---")
print(df.info())

print("\n---Statistical Summary---")
print(df.describe())

print("\nAverage Marks:",df["Marks"].mean())







