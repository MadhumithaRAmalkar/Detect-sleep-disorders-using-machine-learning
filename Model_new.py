import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype  # For handling FutureWarning


# Custom Label Encoder
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


# Load the dataset
try:
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
except FileNotFoundError as e:
    print("File not found. Ensure the dataset is in the correct location.")
    raise e

# Handle missing values
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Normal')

# Encode categorical columns
cat_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
encoders = {col: CustomLabelEncoder().fit(df[col]) for col in cat_cols}

for col in cat_cols:
    df[col] = encoders[col].transform(df[col])

# Save encoders
for col, encoder in encoders.items():
    joblib.dump(encoder, f'{col}_encoder.pkl')

# Split Blood Pressure into Systolic BP and Diastolic BP
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True)
df[['Systolic BP', 'Diastolic BP']] = df[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)
df = df.drop('Blood Pressure', axis=1)

# Ensure the features are the same as the input DataFrame, excluding "Quality of Sleep" and "Daily Steps"
feature_cols = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Physical Activity Level',
                'Stress Level', 'BMI Category', 'Heart Rate', 'Systolic BP', 'Diastolic BP']

df = df[feature_cols + ['Sleep Disorder']]

# Split data into features and target
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the classifiers and their parameter grids
param_grids = {
    'Logistic Regression': {
        'estimator': LogisticRegression(max_iter=1000),
        'param_grid': {'C': [0.1, 1.0, 10.0]}
    },
    'Random Forest': {
        'estimator': RandomForestClassifier(),
        'param_grid': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    },
    # ... (other classifiers omitted for brevity, but unchanged)
}

results = []

for name, config in param_grids.items():
    try:
        grid_search = GridSearchCV(config['estimator'], config['param_grid'], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        mean_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy').mean()
        std_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy').std()
        results.append((name, mean_score, std_score))
        print(f'{name} Best Params: {grid_search.best_params_} Accuracy: {mean_score:.4f} (+/- {std_score:.4f})')
    except Exception as e:
        print(f"Error during GridSearchCV for {name}: {e}")

# Train and evaluate the best classifier on the test set
best_classifier_name = max(results, key=lambda item: item[1])[0]
best_classifier = [config['estimator'] for name, config in param_grids.items() if name == best_classifier_name][0]
best_classifier.fit(X_train, y_train)

train_accuracy = best_classifier.score(X_train, y_train)
test_accuracy = best_classifier.score(X_test, y_test)
print("Best Classifier:", best_classifier_name)
print("Train Accuracy: ", train_accuracy)
print("Test Accuracy: ", test_accuracy)

# Save the best model
joblib.dump(best_classifier, 'best_model.pkl')

# Create a DataFrame for the results
results_df = pd.DataFrame(results, columns=['Model', 'Mean Accuracy', 'Std Dev'])

# Plot the results
plt.figure(figsize=(14, 10))
sns.barplot(x='Mean Accuracy', y='Model', data=results_df, palette='viridis', errorbar=None)
plt.errorbar(results_df['Mean Accuracy'], results_df['Model'], xerr=results_df['Std Dev'], fmt='none', c='black',
             capsize=5)
plt.title('Model Comparison')
plt.xlabel('Mean Accuracy')
plt.ylabel('Model')
plt.legend(title='Model')  # Add legend manually
plt.show()
