import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from datetime import datetime

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    numeric_columns = ['Price', 'Open', 'High', 'Low']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Change %'] = pd.to_numeric(df['Change %'].str.replace('%', ''), errors='coerce')
    
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    return df
    

def calculate_moving_averages(df):
    ma_periods = [50, 200, 365, 500]
    mas = {}
    
    for period in ma_periods:
        ma_name = f'MA_{period}'
        mas[ma_name] = df['Price'].rolling(window=period).mean()
    
    return mas

def fit_arima_model(data):
    diff_data = data['Price'].diff().dropna()
    model = ARIMA(data['Price'], order=(1, 1, 1))
    results = model.fit()
    acf_values = acf(diff_data, nlags=20)
    pacf_values = pacf(diff_data, nlags=20)
    
    return results, acf_values, pacf_values


def analyze_stock_data(csv_path):
    df = prepare_data(csv_path)
    moving_averages = calculate_moving_averages(df)
    arima_results, acf_values, pacf_values = fit_arima_model(df)
    summary = {
        'current_price': df['Price'].iloc[-1],
        'avg_price': df['Price'].mean(),
        'price_std': df['Price'].std(),
        'max_price': df['Price'].max(),
        'min_price': df['Price'].min(),
        'last_ma_values': {name: ma.iloc[-1] for name, ma in moving_averages.items() if not np.isnan(ma.iloc[-1])},
        'arima_aic': arima_results.aic,
        'significant_acf': [i for i, v in enumerate(acf_values) if abs(v) > 0.2]
    }
    
    return df, moving_averages, arima_results, acf_values, pacf_values, summary

def plot_price_and_moving_averages(df, moving_averages):
    plt.figure(figsize=(15, 8)) 
    plt.plot(df.index, df['Price'], label='Price', color='black', alpha=0.7)
    colors = ['blue', 'red', 'green', 'purple']
    for (name, ma), color in zip(moving_averages.items(), colors):
        plt.plot(df.index, ma, label=name, color=color, alpha=0.7)
    
    plt.title('TATA Motors Stock Price and Moving Averages', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True)

    min_price = df['Price'].min()
    max_price = df['Price'].max()
    plt.ylim([min_price * 0.95, max_price * 1.05])  
    plt.tight_layout()
    plt.show()


def plot_acf_values(acf_values):
    plt.figure(figsize=(10, 5))
    lags = range(len(acf_values))
    plt.bar(lags, acf_values)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=0.2, linestyle='--', color='red')
    plt.axhline(y=-0.2, linestyle='--', color='red')
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.grid(True)
    plt.show()


def plot_pacf_values(pacf_values):
    plt.figure(figsize=(10, 5))
    lags = range(len(pacf_values))
    plt.bar(lags, pacf_values)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=0.2, linestyle='--', color='red')
    plt.axhline(y=-0.2, linestyle='--', color='red')
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.grid(True)
    plt.show()



print("Anuj Parihar 21BBS0162\n\n")
df, mas, arima_results, acf_values, pacf_values, summary = analyze_stock_data('tata.csv')

plot_price_and_moving_averages(df, mas)
plot_acf_values(acf_values)
plot_pacf_values(pacf_values)

print("\nAnalysis Summary:")
print(f"Current Price: ₹{summary['current_price']:.2f}")
print(f"Average Price: ₹{summary['avg_price']:.2f}")
print(f"Price Standard Deviation: ₹{summary['price_std']:.2f}")
print(f"Maximum Price: ₹{summary['max_price']:.2f}")
print(f"Minimum Price: ₹{summary['min_price']:.2f}")

print("\nMoving Averages (Latest Values):")
for ma_name, value in summary['last_ma_values'].items():
    print(f"{ma_name}: ₹{value:.2f}")

print("\nARIMA Model Summary:")
print(arima_results.summary())

print("\nSignificant Autocorrelations at lags:", summary['significant_acf'])
