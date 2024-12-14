# xgboost-churn-api
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from joblib import dump

# Step 1: Load Data
data = pd.read_csv('/content/Telco-churn-ml.csv')  # Replace with your dataset path

# Step 2: Preprocess Data
# Target column is 'Churn'
y = data['Churn']
X = data.drop(columns=['Churn'])

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the XGBoost Model
# Define model parameters
params = {
    'objective': 'binary:logistic',  # Binary classification for Churn
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# Train the model
xgb_model = xgb.XGBClassifier(**params)
xgb_model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Step 5: Save the Model
dump(xgb_model, 'xgb_churn_model.joblib')
print("Model saved as 'xgb_churn_model.joblib'")
