import random
import pandas as pd

genders = ['Male', 'Female']
cholesterol_levels = ['Normal', 'High']
physical_activity_levels = ['Low', 'Moderate', 'High']

def generate_random_patient_data(num_rows):
    data = []
    for _ in range(num_rows):
        gender = random.choice(genders)
        age = random.randint(20, 65)  
        weight = random.randint(50, 100)  
        height = random.randint(150, 190)  
        bmi = round(random.uniform(18.0, 35.0), 1) 
        cholesterol = random.choice(cholesterol_levels)
        physical_activity = random.choice(physical_activity_levels)
        
        data.append([gender, age, weight, height, bmi, cholesterol, physical_activity])
    
    df = pd.DataFrame(data, columns=['Gender', 'Age', 'Weight', 'Height', 'BMI', 'Cholesterol', 'PhysicalActivity'])
    
    return df

df = generate_random_patient_data(100)

df.to_csv('2.csv', index=False)
