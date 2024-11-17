import pandas as pd
import numpy as np

def generate_crime_dataset(start_date='2023-01-01', num_days=365):
    np.random.seed(42)
    
    # Create date range with explicit frequency
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    
    base_crimes = 20
    trend = np.linspace(0, 5, num_days)
    seasonal = 5 * np.sin(np.linspace(0, 2*np.pi, num_days))
    noise = np.random.normal(0, 2, num_days)
    
    crime_counts = base_crimes + trend + seasonal + noise
    crime_counts = np.maximum(crime_counts, 0)
    
    df = pd.DataFrame({
        'date': dates,
        'crime_count': np.round(crime_counts),
        'location': 'Delhi'
    })
    
    crime_types = ['Theft', 'Assault', 'Burglary', 'Vandalism', 'Robbery']
    
    crime_type_data = []
    for _, row in df.iterrows():
        total_crimes = int(row['crime_count'])
        type_distribution = np.random.multinomial(total_crimes, 
                                                [0.4, 0.15, 0.2, 0.15, 0.1])
        for crime_type, count in zip(crime_types, type_distribution):
            if count > 0:
                crime_type_data.append({
                    'date': row['date'],
                    'location': row['location'],
                    'crime_type': crime_type,
                    'count': count
                })
    
    detailed_df = pd.DataFrame(crime_type_data)
    detailed_df.to_csv('crime_data.csv', index=False)
    return detailed_df


generate_crime_dataset()