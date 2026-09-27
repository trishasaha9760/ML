import pandas as pd

# User-defined number of students
n = int(input("Enter number of students: "))

names = []
roll_numbers = []
marks = []
attendance = []

# Input student details
for i in range(n):
    print(f"\nEnter details of Student {i + 1}")

    names.append(input("Student Name: "))
    roll_numbers.append(int(input("Roll Number: ")))
    marks.append(float(input("Marks: ")))
    attendance.append(float(input("Attendance (%): ")))

# Create DataFrame
df = pd.DataFrame({
    "Student Name": names,
    "Roll Number": roll_numbers,
    "Marks": marks,
    "Attendance": attendance
})

# Dynamically calculate Grade based on Marks
def calculate_grade(mark):
    if mark >= 90:
        return "A+"
    elif mark >= 80:
        return "A"
    elif mark >= 70:
        return "B"
    elif mark >= 60:
        return "C"
    elif mark >= 50:
        return "D"
    else:
        return "F"

# Add new Grade column
df["Grade"] = df["Marks"].apply(calculate_grade)

# Display DataFrame
print("\nDataFrame with Grade:")
print(df)
