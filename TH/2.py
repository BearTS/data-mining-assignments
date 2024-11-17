import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import warnings
warnings.filterwarnings("ignore")


def calculate_moving_averages(df, window_sizes=[7, 14, 30]):
    if 'date' in df.columns:
        data = df.groupby('date')['count'].sum().reset_index()
        data.set_index('date', inplace=True)
        data.index = pd.DatetimeIndex(data.index, freq='D')
    else:
        data = df
    
    results = pd.DataFrame(index=data.index)
    results['Original'] = data['count']
    
    for window in window_sizes:
        results[f'SMA_{window}'] = data['count'].rolling(window=window).mean()
    
    for window in window_sizes:
        results[f'EMA_{window}'] = data['count'].ewm(span=window, adjust=False).mean()
    
    for window in window_sizes:
        weights = np.arange(1, window + 1)
        results[f'WMA_{window}'] = data['count'].rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum()
        )
    
    for window in window_sizes:
        wma1 = data['count'].rolling(window=window//2).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
        )
        wma2 = data['count'].rolling(window=window).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum()
        )
        results[f'HMA_{window}'] = (2 * wma1 - wma2).rolling(window=int(np.sqrt(window))).mean()
    
    try:
        model = ARIMA(data['count'], order=(1, 0, 1))
        arma_results = model.fit()
        results['ARMA'] = arma_results.fittedvalues
    except:
        print("ARMA calculation failed")
    
    return results

def plot_moving_averages(results, window_sizes=[7, 14, 30]):Lab/DA5/DA5.docx
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=results['Original'], label='Original', alpha=0.5, color='gray')
    for window in window_sizes:
        sns.lineplot(data=results[f'SMA_{window}'], label=f'SMA {window} days')
    plt.title('Simple Moving Averages', pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=results['Original'], label='Original', alpha=0.5, color='gray')
    for window in window_sizes:
        sns.lineplot(data=results[f'EMA_{window}'], label=f'EMA {window} days')
    plt.title('Exponential Moving Averages', pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=results['Original'], label='Original', alpha=0.5, color='gray')
    for window in window_sizes:
        sns.lineplot(data=results[f'WMA_{window}'], label=f'WMA {window} days')
    plt.title('Weighted Moving Averages', pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=results['Original'], label='Original', alpha=0.5, color='gray')
    for window in window_sizes:
        sns.lineplot(data=results[f'HMA_{window}'], label=f'HMA {window} days')
    plt.title('Hull Moving Averages', pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=results['Original'], label='Original', alpha=0.5, color='gray')
    sns.lineplot(data=results['SMA_7'], label='SMA 7 days')
    sns.lineplot(data=results['SMA_30'], label='SMA 30 days')
    
    crosses = np.where(np.diff(np.signbit(
        results['SMA_7'].fillna(0) - results['SMA_30'].fillna(0))))[0]
    
    for cross in crosses:
        plt.plot(results.index[cross], results['Original'][cross], 
                'ro' if results['SMA_7'][cross] > results['SMA_30'][cross] else 'go',
                alpha=0.6)
    
    plt.title('Moving Average Crossover (7 and 30 days)', pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if 'ARMA' in results.columns:
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=results['Original'], label='Original', alpha=0.5, color='gray')
        sns.lineplot(data=results['ARMA'], label='ARMA', color='red')
        plt.title('ARMA (Autoregressive Moving Average)', pad=20)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # 7. Comparison of All Methods
    plt.figure(figsize=(15, 8))
    # Create a long-format DataFrame for seaborn
    comparison_data = pd.DataFrame({
        'Original': results['Original'],
        'SMA': results['SMA_14'],
        'EMA': results['EMA_14'],
        'WMA': results['WMA_14'],
        'HMA': results['HMA_14']
    })
    if 'ARMA' in results.columns:
        comparison_data['ARMA'] = results['ARMA']
    
    # Plot using seaborn
    for column in comparison_data.columns:
        sns.lineplot(data=comparison_data[column], label=column)
    
    plt.title('Comparison of Different Moving Averages (14-day period)', pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("Anuj Parihar 21BBS0162\n\n")
sns.set_theme(style="whitegrid", font_scale=1.2)
df = pd.read_csv('crime_data.csv')
df['date'] = pd.to_datetime(df['date'])

results = calculate_moving_averages(df)

plot_moving_averages(results)

summary_stats = results.agg(['mean', 'std', 'min', 'max']).round(2)
print(summary_stats)