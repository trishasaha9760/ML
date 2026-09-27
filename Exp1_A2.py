import pandas as pd

# Create empty lists
names = []
roll_numbers = []
marks = []
attendance = []

# Get number of students
n = int(input("Enter the number of students: "))

# Get student details from user
for i in range(n):
    print(f"\nEnter details of Student {i + 1}")

    name = input("Student Name: ")
    roll = int(input("Roll Number: "))
    mark = float(input("Marks: "))
    attend = float(input("Attendance (%): "))

    names.append(name)
    roll_numbers.append(roll)
    marks.append(mark)
    attendance.append(attend)

# Create DataFrame
df = pd.DataFrame({
    "Student Name": names,
    "Roll Number": roll_numbers,
    "Marks": marks,
    "Attendance": attendance
})

print("\nComplete DataFrame:")
print(df)

# Filter students scoring above 80
filtered_df = df[df["Marks"] > 80]

print("\nStudents who scored above 80 marks:")
print(filtered_df)