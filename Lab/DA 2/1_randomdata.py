import numpy as np
import pandas as pd

np.random.seed(42)

genders = ['Male', 'Female']
cholesterol_levels = ['Normal', 'High']
physical_activity_levels = ['Low', 'Moderate', 'High']
strict_diet = ['Yes', 'No']
data = {
    'Gender': np.random.choice(genders, size=100),
    'Age': np.random.randint(20, 65, size=100), 
    'Weight': np.random.randint(50, 101, size=100), 
    'Height': np.random.randint(150, 191, size=100), 
    'BMI': np.round(np.random.uniform(18.5, 35.0, size=100), 1),
    'Cholesterol': np.random.choice(cholesterol_levels, size=100),
    'PhysicalActivity': np.random.choice(physical_activity_levels, size=100),
    'StrictDietRequired': np.random.choice(strict_diet, size=100)
}

df = pd.DataFrame(data)

df.to_csv('1.csv', index=False)