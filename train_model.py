import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
import yfinance as yf
import datetime
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configuration ---
TICKER = 'SPY' # S&P 500 ETF or any relevant financial instrument
DAYS_TO_LOOK_BACK = 365 * 3 # 3 years of data
TEST_SIZE = 0.2
RANDOM_SEED = 42

# --- 1. Data Acquisition and Preparation ---

def fetch_and_prepare_data(ticker, days_lookback):
    """Fetches data and prepares basic technical features and the target variable."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days_lookback)

    print(f"Fetching {ticker} data from {start_date} to {end_date}...")

    # Fetch data from Yahoo Finance
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None

    if data.empty:
        print("Error: Could not fetch data. Check the ticker symbol.")
        return None, None, None

    # Feature Engineering
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI_14'] = data['Close'].diff()
    data['RSI_14'] = data['RSI_14'].apply(lambda x: x if x > 0 else 0).rolling(window=14).mean() / data['RSI_14'].abs().rolling(window=14).mean()

    # Daily Return (Volatility Proxy)
    data['Daily_Return'] = data['Close'].pct_change()

    # Target Definition: 1 if the next day's price is higher, 0 otherwise
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Clean up NaN values created by rolling windows and the shift operation
    data.dropna(inplace=True)

    # Define features and target
    features = ['SMA_20', 'RSI_14', 'Daily_Return']
    X = data[features].values
    y = data['Target'].values

    return X, y, features

# --- 2. Training Workflow ---

X, y, features = fetch_and_prepare_data(TICKER, DAYS_TO_LOOK_BACK)

if X is not None and len(X) > 0:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_SEED
    )

    # --- 3. Instantiate and Fit Scaler (The Fix for 'scaler' not defined) ---
    print("\nScaling features...")
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit the scaler ONLY on the training data and transform both sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. Instantiate and Train Model (The Fix for 'model' not defined) ---
    print("Training Logistic Regression model...")
    # Initialize the model
    model = LogisticRegression(solver='liblinear', random_state=RANDOM_SEED)
    # Train the model on the scaled training data
    model.fit(X_train_scaled, y_train)

    # Evaluate the model (Optional, but good practice)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Test Accuracy: {accuracy:.4f}")
    print(f"Features used: {', '.join(features)}")

    # -------------------------------------------------------
    # 5. Save the trained model and scaler (Original section)
    # -------------------------------------------------------

    # The previous line: os.makedirs(os.path.dirname(''), exist_ok=True)
    # was causing an error because os.path.dirname('') returns an empty string.
    # We remove it as joblib.dump will save to the current directory by default.

    joblib.dump(scaler, 'risk_scaler.pkl')
    joblib.dump(model, 'risk_model.pkl')

    print("\n✅ Model training complete.")
    print("Files 'risk_model.pkl' and 'risk_scaler.pkl' saved successfully!")
    print("You can now run the AI agent (app.py) without requiring any local dataset.")

else:
    # This path handles the case where data acquisition failed, so we don't try to save undefined variables
    print("\nTraining aborted due to data fetching or preparation error.")
    # Define placeholder variables to prevent Pylance errors in the IDE, though they won't be saved
    scaler = None
    model = None