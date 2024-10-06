import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

print("Anuj Parihar 21BBS0162\n\n")

data = pd.read_csv('1.csv')

def classify_student(row):
    if row['AttendancePercentage'] < 80 or row['AverageGrade'] < 60:
        return 'At Risk'
    return 'Not At Risk'

data['Classification'] = data.apply(classify_student, axis=1)

X = data[['AttendancePercentage', 'AverageGrade', 'ExtracurricularActivities', 'StudyHoursPerWeek']]
y = data['Classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

all_students_scaled = scaler.transform(X)
all_predictions = knn.predict(all_students_scaled)

data['Predicted_Classification'] = all_predictions

print("\nStudents classified as 'At Risk':")
at_risk_students = data[data['Predicted_Classification'] == 'At Risk']
print(at_risk_students[['StudentID', 'Name', 'AttendancePercentage', 'AverageGrade', 'Predicted_Classification']])

print("\nStudents classified as 'Not At Risk':")
not_at_risk_students = data[data['Predicted_Classification'] == 'Not At Risk']
print(not_at_risk_students[['StudentID', 'Name', 'AttendancePercentage', 'AverageGrade', 'Predicted_Classification']])

