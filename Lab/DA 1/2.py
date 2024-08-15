import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("Made by Anuj Parihar 21BBS0162")
data = pd.read_csv('Fuel.csv')
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Set style for better-looking plots
plt.style.use('ggplot')  # Changed from 'seaborn' to 'ggplot'

# 1. Line Plot
plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['Delhi'], marker='o')
plt.title('Delhi Data Over Years')
plt.xlabel('Year')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# 2. Bar Plot
plt.figure(figsize=(12, 6))
plt.bar(data['Year'].dt.year, data['Delhi'])
plt.title('Delhi Data Over Years')
plt.xlabel('Year')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

# 3. Scatter Plot
plt.figure(figsize=(12, 6))
plt.scatter(data['Year'], data['Delhi'])
plt.title('Delhi Data Over Years')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()

# 4. Area Plot
plt.figure(figsize=(12, 6))
plt.fill_between(data['Year'], data['Delhi'])
plt.title('Delhi Data Over Years')
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()

# 5. Box Plot
data['Decade'] = (data['Year'].dt.year // 10) * 10
plt.figure(figsize=(12, 6))
sns.boxplot(x='Decade', y='Delhi', data=data)
plt.title('Distribution of Delhi Data by Decade')
plt.xlabel('Decade')
plt.ylabel('Value')
plt.show()

# 6. Heatmap
data['Year_num'] = data['Year'].dt.year
data_pivot = data.pivot(index='Year_num', columns='Year_num', values='Delhi')
changes = data_pivot.diff(axis=1)

plt.figure(figsize=(12, 8))
sns.heatmap(changes, cmap='coolwarm', center=0, annot=True, fmt='.2f')
plt.title('Year-to-Year Changes in Delhi Data')
plt.xlabel('Year')
plt.ylabel('Year')
plt.show()

# 7. Histogram
plt.figure(figsize=(12, 6))
plt.hist(data['Delhi'], bins=10, edgecolor='black')
plt.title('Distribution of Delhi Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

print("All charts have been displayed as popups.")
