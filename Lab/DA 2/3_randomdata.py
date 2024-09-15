import random
import pandas as pd
# Parameters for generating random data
NUM_RECORDS = 50
ROAD_IDS = list(range(101, 101 + NUM_RECORDS))
LENGTH_RANGE = (2.0, 10.0)  
BENDS_RANGE = (3, 20) 
TRAFFIC_VOLUMES = ['Low', 'Medium', 'High']
ACCIDENT_RISKS = ['Low', 'Medium', 'High']

def random_traffic_volume():
    return random.choice(TRAFFIC_VOLUMES)

def random_accident_risk(traffic_volume):
    if traffic_volume == 'High':
        return random.choices(ACCIDENT_RISKS, weights=[0.2, 0.3, 0.5])[0]  # More likely to have a high accident risk
    elif traffic_volume == 'Medium':
        return random.choices(ACCIDENT_RISKS, weights=[0.3, 0.4, 0.3])[0]
    else:  
        return random.choices(ACCIDENT_RISKS, weights=[0.5, 0.3, 0.2])[0]

data = {
    'RoadID': ROAD_IDS,
    'Length': [round(random.uniform(*LENGTH_RANGE), 1) for _ in range(NUM_RECORDS)],
    'NumberofBends': [random.randint(*BENDS_RANGE) for _ in range(NUM_RECORDS)],
    'Trafficvolume': [],
    'AccidentRisk': []
}

for _ in range(NUM_RECORDS):
    traffic_volume = random_traffic_volume()
    accident_risk = random_accident_risk(traffic_volume)
    data['Trafficvolume'].append(traffic_volume)
    data['AccidentRisk'].append(accident_risk)

df = pd.DataFrame(data)
df.to_csv('3.csv', index=False)