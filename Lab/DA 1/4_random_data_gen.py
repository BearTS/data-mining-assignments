# generate_sample_data.py

import csv
import random

def generate_sample_data(num_transactions, items, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['transaction_id', 'items'])
        
        for i in range(num_transactions):
            num_items = random.randint(1, 5)
            transaction_items = ', '.join(random.sample(items, num_items))
            writer.writerow([i+1, transaction_items])


num_transactions = 50
items = [
    'Sedan', 'SUV', 'Hatchback', 'Truck', 'Van',
    'Sport Car', 'Luxury Car', 'Electric Car', 'Hybrid Car',
    'Extended Warranty', 'Insurance', 'Financing',
    'Maintenance Package', 'Tire Package', 'Paint Protection',
    'GPS Navigation', 'Audio System Upgrade', 'Leather Seats',
    'Sunroof', 'Towing Package'
]
file_path = 'monthly_sales.csv'

generate_sample_data(num_transactions, items, file_path)
print(f"Sample data generated and saved to {file_path}")


