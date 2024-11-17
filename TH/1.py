import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

def analyze_crime_data(df, crime_type='all', forecast_days=30):
    if crime_type != 'all':
        data = df[df['crime_type'] == crime_type].groupby('date')['count'].sum()
    else:
        data = df.groupby('date')['count'].sum()
    
    data.index = pd.DatetimeIndex(data.index, freq='D')
    
    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = data[train_size:]
    
    arima_model = ARIMA(train, order=(2,1,2), freq='D')
    arima_results = arima_model.fit()
    
    arima_forecast = arima_results.forecast(steps=len(test))
    
    ar_model = AutoReg(train, lags=7)
    ar_results = ar_model.fit()
    
    ar_forecast = ar_results.predict(start=len(train), end=len(data)-1)
    
    arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
    ar_rmse = np.sqrt(mean_squared_error(test, ar_forecast))
    
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='D')[1:]
    future_forecast = arima_results.forecast(steps=forecast_days)
    plt.figure(figsize=(15, 8))
    plt.plot(train.index, train, label='Training Data', color='black')
    plt.plot(test.index, test, label='Test Data', color='gray')
    plt.plot(test.index, arima_forecast, 
             label=f'ARIMA Forecast (RMSE: {arima_rmse:.2f})', 
             color='red', linestyle='--')
    plt.plot(test.index, ar_forecast, 
             label=f'AR Forecast (RMSE: {ar_rmse:.2f})', 
             color='blue', linestyle='--')
    
    plt.plot(future_dates, future_forecast, 
             label='Future Forecast', 
             color='green', linestyle=':')
    
    plt.title(f'Crime Trend Analysis and Forecast ({crime_type})')
    plt.xlabel('Date')
    plt.ylabel('Number of Incidents')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return {
        'arima_rmse': arima_rmse,
        'ar_rmse': ar_rmse,
        'future_forecast': future_forecast,
        'model': arima_results
    }

print("Anuj Parihar 21BBS0162\n\n")
df = pd.read_csv('crime_data.csv')
results = analyze_crime_data(df)

theft_results = analyze_crime_data(df, crime_type='Theft')

print("\nSummary Statistics:")
print(f"Overall ARIMA RMSE: {results['arima_rmse']:.2f}")
print(f"Overall AR RMSE: {results['ar_rmse']:.2f}")
print(f"Theft ARIMA RMSE: {theft_results['arima_rmse']:.2f}")
print(f"Theft AR RMSE: {theft_results['ar_rmse']:.2f}")

plt.show()