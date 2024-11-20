import pandas as pd
import joblib
from Model_new import CustomLabelEncoder  # Import the CustomLabelEncoder class

# Load the encoders and model
cat_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
encoders = {col: joblib.load(f'{col}_encoder.pkl') for col in cat_cols}
xgb = joblib.load('xgb_model.pkl')

def preprocess_input(input_df):
    # Encode categorical columns
    for col in cat_cols[:-1]:  # Exclude 'Sleep Disorder'
        input_df[col] = encoders[col].transform(input_df[col])

    # Split Blood Pressure into Systolic BP and Diastolic BP
    input_df[['Systolic BP', 'Diastolic BP']] = input_df['Blood Pressure'].str.split('/', expand=True)
    input_df[['Systolic BP', 'Diastolic BP']] = input_df[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)
    input_df = input_df.drop('Blood Pressure', axis=1)

    # Ensure the input DataFrame has the same columns as the training DataFrame
    feature_cols = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
                    'Physical Activity Level', 'Stress Level', 'BMI Category', 'Heart Rate',
                    'Daily Steps', 'Systolic BP', 'Diastolic BP']
    input_df = input_df[feature_cols]

    return input_df

def predict_sleep_disorder(input_df):
    processed_df = preprocess_input(input_df)
    predictions = xgb.predict(processed_df)
    return encoders['Sleep Disorder'].inverse_transform(predictions)
