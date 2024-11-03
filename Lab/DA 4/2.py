import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

print("Anuj Parihar 21BBS0162\n\n")

df = pd.read_csv('2.csv')

label_encoders = {}
for column in ['Income', 'Criminal Record', 'EXP Load', 'Approved?']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df[['Income', 'Criminal Record', 'EXP Load']]
y = df['Approved?']

model = GaussianNB()
model.fit(X, y)

test_case = pd.DataFrame({
    'Income': [label_encoders['Income'].transform(['30-70'])[0]],
    'Criminal Record': [label_encoders['Criminal Record'].transform(['Yes'])[0]],
    'EXP Load': [label_encoders['EXP Load'].transform(['>5'])[0]]
})

probability = model.predict_proba(test_case)

print(f"Probability of loan approval (No, Yes): {probability[0]}")
print(f"Probability of loan approval (Yes): {probability[0][1]:.2f}")
