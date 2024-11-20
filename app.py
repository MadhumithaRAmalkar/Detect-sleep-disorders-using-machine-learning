from flask import Flask, request, render_template
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/sleep_health')
def sleep_health():
    return render_template('sleep_health.html')

@app.route('/')
def index():
    return render_template('index.html')

class CustomLabelEncoder(LabelEncoder):
    def fit(self, y):
        super().fit(y)
        return self

    def transform(self, y):
        unseen_labels = set(y) - set(self.classes_)
        if unseen_labels:
            print(f"Warning: Unseen labels encountered: {unseen_labels}")
        return super().transform([label if label in self.classes_ else self.classes_[0] for label in y])
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)

# Load encoders, scaler, and model
encoders = {
    'Gender': joblib.load('Gender_encoder.pkl'),
    'Occupation': joblib.load('Occupation_encoder.pkl'),
    'BMI Category': joblib.load('BMI Category_encoder.pkl'),
    'Sleep Disorder': joblib.load('Sleep Disorder_encoder.pkl')
}

scaler = joblib.load('scaler.pkl')
model = joblib.load('best_model.pkl')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        try:
            # Retrieve form data with default values for missing fields
            data = {
                'Gender': request.form.get('Gender', 'Unknown'),
                'Age': float(request.form.get('Age', 0)),
                'Occupation': request.form.get('Occupation', 'Unknown'),
                'Sleep Duration': float(request.form.get('Sleep Duration', 0)),
                'Physical Activity Level': float(request.form.get('Physical Activity Level', 0)),
                'Stress Level': float(request.form.get('Stress Level', 0)),
                'BMI Category': request.form.get('BMI Category', 'Unknown'),
                'Heart Rate': float(request.form.get('Heart Rate', 0)),
                'Systolic BP': float(request.form.get('Systolic BP', 0)),
                'Diastolic BP': float(request.form.get('Diastolic BP', 0))
            }
            
            # Create DataFrame for the input
            df_input = pd.DataFrame([data])
            
            # Encode categorical columns
            for col in ['Gender', 'Occupation', 'BMI Category']:
                df_input[col] = encoders[col].transform(df_input[col])
            
            # Feature scaling
            X_scaled = scaler.transform(df_input)
            
            # Predict
            prediction = model.predict(X_scaled)
            prediction = encoders['Sleep Disorder'].inverse_transform(prediction)[0]
            
            return render_template('result.html', prediction=prediction)
        
        except Exception as e:
            print(f"Error: {e}")
            return render_template('result.html', prediction="Error in prediction.")
    
    return render_template('result.html', prediction=None)

if _name_ == '_main_':
    # Ensuring the app runs on the correct host and port provided by Render
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
