import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(df):
    X = df.drop('Admitted', axis=1)
    y = df['Admitted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    return model, y_pred, feature_importance

def plot_results(model, X_test, y_test, y_pred, feature_importance):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=axes[0,1])
    axes[0,1].set_title('Feature Importance')
    
    probabilities = model.predict_proba(X_test)[:, 1]
    sns.histplot(probabilities, bins=30, ax=axes[1,0])
    axes[1,0].set_title('Probability Distribution')
    axes[1,0].set_xlabel('Probability of Admission')
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    axes[1,1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1,1].plot([0, 1], [0, 1], 'k--')
    axes[1,1].set_title('ROC Curve')
    axes[1,1].set_xlabel('False Positive Rate')
    axes[1,1].set_ylabel('True Positive Rate')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('admission_analysis.png')
    plt.close()

def predict_admission(model, scaler, student_data):
    scaled_data = scaler.transform(student_data)
    prob = model.predict_proba(scaled_data)[0][1]
    
    return prob

df = pd.read_csv('2.csv')
print("\nSample data shape:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())

X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)

feature_names = df.columns[:-1] 
model, y_pred, feature_importance = train_evaluate_model(
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nFeature Importance:")
print(feature_importance)

print("\nGenerating plots...")
plot_results(model, X_test_scaled, y_test, y_pred, feature_importance)

example_student = pd.DataFrame({
    'CGPA': [9.0],
    'GRE_Score': [320],
    'TOEFL_Score': [110],
    'Research_Papers': [2],
    'Mini_Projects': [4],
    'Internships': [2]
})

prob = predict_admission(model, scaler, example_student)
print(f"\nProbability of admission of a sample new student: {prob:.2%}")
