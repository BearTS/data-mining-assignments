import csv
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Define the number of students
num_students = 100

# Define the headers for our CSV
headers = ['StudentID', 'Name', 'AttendancePercentage', 'AverageGrade', 'ExtracurricularActivities', 'StudyHoursPerWeek']

# Generate random data
data = []
for i in range(num_students):
    student_id = i + 1
    name = fake.name()
    attendance = round(random.uniform(60, 100), 2)
    grade = round(random.uniform(40, 100), 2)
    extracurricular = random.randint(0, 5)
    study_hours = random.randint(0, 40)
    
    data.append([student_id, name, attendance, grade, extracurricular, study_hours])

# Write data to CSV file
with open('1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data)

print("Random student data has been generated and saved to '1.csv'")