# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('employee.csv')


# 1. Get the dimensions, structure, attribute name, and attribute values of the dataset
print("Made by Anuj Parihar 21BBS0162")
print("1. Dataset Information:")
print("\nDimensions:")
print(df.shape)

print("\nStructure:")
print(df.info())

print("\nAttribute names:")
print(df.columns)

print("\nAttribute values:")
print(df.describe())

# 2. Display various parts of the dataset
print("\n")
print("Made by Anuj Parihar 21BBS0162")
print("\n2. Displaying parts of the dataset:")

print("\n(A) First 5 Records:")
print(df.head())

print("\n(B) Last 5 Records:")
print(df.tail())

print("\n(C) Name, Designation, Salary of First 10 records:")
print(df[['Name', 'Designation', 'Salary']].head(10))

print("\n(D) Name of all records:")
print(df['Name'])

print("\n(E) All records:")
print(df)

# 3. Display statistical measures of the dataset
print("\n")
print("Made by Anuj Parihar 21BBS0162")
print("\n3. Statistical measures of the dataset:")

print("\na) Mean, median and mode of the variables:")
print("Mean:")
print(df.mean(numeric_only=True))
print("\nMedian:")
print(df.median(numeric_only=True))
print("\nMode:")
print(df.mode())

print("\nb) Variance and Covariance:")
print("Variance:")
print(df.var(numeric_only=True))
print("\nCovariance:")
print(df.cov(numeric_only=True))

print("\nc) Correlation of salary to experience:")
print(df['Salary'].corr(df['Experience']))
# 4. Draw charts
print("\n")
print("Made by Anuj Parihar 21BBS0162")
print("\n4. Drawing charts:")

# A) Pie chart on designation
plt.figure(figsize=(10, 6))
df['Designation'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Pie Chart of Designations')
plt.ylabel('')
plt.show()

# B) Histogram of Salary
plt.figure(figsize=(10, 6))
plt.hist(df['Salary'], bins=10, edgecolor='black')
plt.title('Histogram of Salary')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# C) Scatter plot of Salary to Experience
plt.figure(figsize=(10, 6))
plt.scatter(df['Experience'], df['Salary'])
plt.title('Scatter Plot of Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
