import pandas as pd
import numpy as np

def create_sample_data(n_samples=200):
    np.random.seed(42)
    
    # Generate random data
    data = {
        'CGPA': np.random.uniform(6.0, 10.0, n_samples),
        'GRE_Score': np.random.randint(260, 340, n_samples),
        'TOEFL_Score': np.random.randint(80, 120, n_samples),
        'Research_Papers': np.random.randint(0, 5, n_samples),
        'Mini_Projects': np.random.randint(1, 6, n_samples),
        'Internships': np.random.randint(0, 4, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate probability based on weighted factors
    prob = (
        0.3 * (df['CGPA'] - 6) / 4 +  # CGPA has 30% weight
        0.25 * (df['GRE_Score'] - 260) / 80 +  # GRE has 25% weight
        0.15 * (df['TOEFL_Score'] - 80) / 40 +  # TOEFL has 15% weight
        0.1 * df['Research_Papers'] / 4 +  # Research has 10% weight
        0.1 * df['Mini_Projects'] / 5 +  # Projects have 10% weight
        0.1 * df['Internships'] / 3  # Internships have 10% weight
    )
    
    # Convert probability to binary outcome (Admitted: 1, Not Admitted: 0)
    df['Admitted'] = (prob + np.random.normal(0, 0.1, n_samples) > 0.6).astype(int)
    
    # Save to CSV
    df.to_csv('2.csv', index=False)
    return df


create_sample_data(100)