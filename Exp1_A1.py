import numpy as np

# User enters marks
marks = []

for i in range(10):
    mark = float(input(f"Enter marks of student {i+1}: "))
    marks.append(mark)

# Convert list to NumPy array
marks = np.array(marks)

# Calculate and print statistics
print("\nMarks:", marks)
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Standard Deviation:", np.std(marks))
print("Maximum:", np.max(marks))
print("Minimum:", np.min(marks))
