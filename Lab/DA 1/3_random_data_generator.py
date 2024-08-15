import random
import pandas as pd

# List of juice items
juices = ['Apple Juice', 'Orange Juice', 'Mango Juice', 'Grape Juice', 'Pineapple Juice', 
          'Cranberry Juice', 'Pomegranate Juice', 'Lemon Juice', 'Strawberry Juice', 'Tomato Juice']

# Function to generate a random juice combination
def generate_juice_combination():
    num_juices = random.randint(1, 3)  # Random number of juices (1 to 3)
    return ','.join(random.sample(juices, num_juices))

# Generate random data
num_transactions = 20  # You can change this to generate more or fewer transactions
data = {
    'transaction_id': list(range(1, num_transactions + 1)),
    'item': [generate_juice_combination() for _ in range(num_transactions)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Optionally, save to CSV
df.to_csv('juice_transactions.csv', index=False)
print("Data saved to 'juice_transactions.csv'")
