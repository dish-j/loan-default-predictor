# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv('Loan_Default.csv')

# Basic cleaning
df.drop(['ID'], axis=1, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# ✅ Select only numeric features matching your HTML form inputs
X = df[['loan_amount', 'income', 'Credit_Score']]
y = df['Status']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model with class weight balanced to handle class imbalance
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'model/loan_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Model and scaler saved successfully.")
