import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
print('Anuj Parihar 21BBS0162')
data = pd.read_csv('3.csv')
label_encoders = {}
for column in ['Trafficvolume','AccidentRisk']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
features = ['Length', 'NumberofBends', 'Trafficvolume']
target = 'AccidentRisk'
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
tree_rules = export_text(clf, feature_names=features)
print("Decision Tree Rules:")
print(tree_rules)