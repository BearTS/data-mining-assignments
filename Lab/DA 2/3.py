import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

print('Anuj Parihar 21BBS0162')

# Load data
data = pd.read_csv('3.csv')

# Encode categorical columns
label_encoders = {}
for column in ['Trafficvolume', 'AccidentRisk']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features and target
features = ['Length', 'Numberof_Bends', 'Trafficvolume']
target = 'AccidentRisk'
X = data[features]
y = data[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Print classification report and accuracy
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

# Print decision tree rules as text
tree_rules = export_text(clf, feature_names=features)
print("Decision Tree Rules:")
print(tree_rules)

# Plot and display decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=features, class_names=label_encoders['AccidentRisk'].classes_, filled=True)
plt.show()