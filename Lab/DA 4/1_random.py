import pandas as pd
import numpy as np
import random
from faker import Faker

num_records = 50
locations = ['Vellore', 'Mumbai', 'Delhi', 'Indore', 'Bengaluru']

fake = Faker()

# Generate the dataset
data = {
    'Name': [fake.name() for _ in range(num_records)], 
    'Location': [random.choice(locations) for _ in range(num_records)],
    'Height': np.random.normal(loc=170, scale=10, size=num_records).astype(int),  
    'Weight': np.random.normal(loc=70, scale=15, size=num_records).astype(int),   
    'Age': np.random.randint(18, 60, size=num_records)  
}

df = pd.DataFrame(data)

csv_file = 'user_data.csv'
df.to_csv(csv_file, index=False)
