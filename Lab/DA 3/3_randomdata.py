import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for dataset
num_entries = 100  # Number of entries in the dataset

# Randomly generate data
weights = np.random.randint(80, 200, size=num_entries)  # Random weights between 80 and 200 grams
color_scores = np.round(np.random.uniform(0.25, 0.95, size=num_entries), 2)  # Random color scores
sugar_contents = np.random.randint(3, 13, size=num_entries)  # Random sugar content between 3 and 12 grams

# Fruit types
fruit_types = ['Apple', 'Banana', 'Orange', 'Grape']

# Randomly select fruit types
selected_fruit_types = np.random.choice(fruit_types, size=num_entries)

# Create DataFrame
df = pd.DataFrame({
    'Weight': weights,
    'Color_Score': color_scores,
    'Sugar_Content': sugar_contents,
    'Fruit_Type': selected_fruit_types
})

# Save to CSV
df.to_csv('3.csv', index=False)

