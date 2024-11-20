import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Handle missing values
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Normal  ')

# Encode categorical columns
cat_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}

for col in cat_cols:
    df[col] = encoders[col].transform(df[col])

# Save the encoders
for col, encoder in encoders.items():
    joblib.dump(encoder, f'{col}_encoder.pkl')

# Split Blood Pressure into Systolic BP and Diastolic BP
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic BP', 'Diastolic BP']] = df[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)
df = df.drop('Blood Pressure', axis=1)

# Ensure the features are the same as the input DataFrame
feature_cols = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
                'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate',
                'Daily Steps', 'Systolic BP', 'Diastolic BP']

df = df[feature_cols + ['Sleep Disorder']]

# Split data into features and target
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBClassifier
xgb = XGBClassifier().fit(X_train, y_train)

# Evaluate the model
train_accuracy = xgb.score(X_train, y_train)
test_accuracy = xgb.score(X_test, y_test)
print("Train Accuracy: ", train_accuracy)
print("Test Accuracy: ", test_accuracy)

# Save the model
joblib.dump(xgb, 'xgb_model.pkl')
