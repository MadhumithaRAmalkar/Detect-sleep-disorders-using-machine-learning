import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_model_and_predict(model_path, scaler_path, new_data):
    """
    Load a model and a scaler from .pkl files and make predictions on new data.

    Parameters:
    - model_path (str): Path to the .pkl file containing the trained model.
    - scaler_path (str): Path to the .pkl file containing the fitted scaler.
    - new_data (pd.DataFrame): Data for which predictions are to be made. Should have the same features as the training data.

    Returns:
    - predictions (np.ndarray): Predicted labels for the new data.
    """
    try:
        # Load the model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Check if new_data is a DataFrame
        if not isinstance(new_data, pd.DataFrame):
            raise ValueError("new_data must be a pandas DataFrame")

        # Ensure all columns are properly formatted
        new_data = new_data.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

        # Feature scaling
        X_new_scaled = scaler.transform(new_data)

        # Make predictions
        predictions = model.predict(X_new_scaled)

        return predictions

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Sample data for testing
    sample_data = pd.DataFrame({
        'Gender': [1, 0],
        'Age': [25, 40],
        'Occupation': [2, 1],
        'Sleep Duration': [7, 6],
        'Physical Activity Level': [3, 2],
        'Stress Level': [4, 3],
        'BMI Category': [1, 0],
        'Heart Rate': [70, 80],
        'Systolic BP': [120, 140],
        'Diastolic BP': [80, 90]
    })

    # Path to the saved model and scaler
    model_path = 'best_model.pkl'
    scaler_path = 'scaler.pkl'  # Ensure you have saved the scaler previously

    # Load model and predict
    predictions = load_model_and_predict(model_path, scaler_path, sample_data)

    print("Predictions:", predictions)
