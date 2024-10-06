import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

print("\nAnuj Parihar 21BBS0162\n")
df = pd.read_csv('3.csv')

le = LabelEncoder()
df['Fruit_Type'] = le.fit_transform(df['Fruit_Type'])

X = df[['Weight', 'Color_Score', 'Sugar_Content']]
y = df['Fruit_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
