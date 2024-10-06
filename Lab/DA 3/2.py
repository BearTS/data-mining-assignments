import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

print("Anuj Parihar 21BBS0162\n\n")

data = pd.read_csv('2.csv')

data['Month'] = pd.to_datetime(data['Month'])
data['MonthNum'] = (data['Month'].dt.year - 2019) * 12 + data['Month'].dt.month

X = data['MonthNum'].values.reshape(-1, 1)
y = data['DEMAT_Count'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse}")
print(f"R-squared score: {r2}")

jan_2025 = (2025 - 2019) * 12 + 1  # Convert January 2025 to numeric representation
predicted_count = model.predict([[jan_2025]])[0]

print(f"\nPredicted DEMAT account count for January 2025: {predicted_count:,.0f}")
plt.figure(figsize=(12, 6))
plt.scatter(data['Month'], data['DEMAT_Count'], color='blue', label='Actual Data')
plt.plot(data['Month'], model.predict(X), color='red', label='Linear Regression')

future_date = pd.to_datetime('2025-01-01')
plt.scatter(future_date, predicted_count, color='green', s=100, label='January 2025 Prediction')

plt.title('DEMAT Account Count Prediction')
plt.xlabel('Date')
plt.ylabel('DEMAT Account Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.figtext(0.99, 0.01, 'Anuj Parihar 21BBS0162', horizontalalignment='right')
plt.show()