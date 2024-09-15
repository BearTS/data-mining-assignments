import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
print('Anuj Parihar 21BBS0162')
data = pd.read_csv('2.csv')
label_encoders = {}
for column in ['Gender', 'Cholesterol', 'PhysicalActivity']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)
data['Cluster'] = kmeans.labels_
cluster_mapping = {0: 'Normal' , 1: 'Healthy' , 2: 'Weak'}
data['Cluster'] = data['Cluster'].map(cluster_mapping)
data.to_csv('2.csv', index=False)
print(data.head())

plt.scatter(data['Age'], data['BMI'], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('K-means Clustering of Patients')
plt.show()