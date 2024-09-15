import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
print('Anuj Parihar 21BBS0162')
data = pd.read_csv('1.csv')
label_encoders = {}
for column in ['Gender' , 'Cholesterol', 'PhysicalActivity', 'StrictDietRequired']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
X = data.drop('StrictDietRequired', axis=1)
y = data['StrictDietRequired']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print()
print(f'Accuracy: {accuracy * 100} %')
print('Classification Report:')
print(report)