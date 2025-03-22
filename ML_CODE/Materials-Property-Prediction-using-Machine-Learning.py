import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load Dataset (Replace with actual dataset file)
df = pd.read_csv('materials_dataset.csv')  # Assume dataset contains composition & properties

# Feature Engineering (Selecting Numerical Features for Training)
X = df.drop(columns=['Yield_Strength'])  # Drop target variable
y = df['Yield_Strength']  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train Models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Neural Network Model (GNN Placeholder for Simplicity)
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test).flatten()

# Performance Evaluation
def evaluate_model(y_true, y_pred, model_name):
    print(f'\n{model_name} Performance:')
    print(f'MAE: {mean_absolute_error(y_true, y_pred):.4f}')
    print(f'R² Score: {r2_score(y_true, y_pred):.4f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}')

evaluate_model(y_test, y_pred_rf, 'Random Forest')
evaluate_model(y_test, y_pred_xgb, 'XGBoost')
evaluate_model(y_test, y_pred_nn, 'Neural Network')
