"""
Prepare dataset for startup valuation modeling.

Steps:
- Load CSV: Startup Growth and Funding Trends.csv
- Select relevant columns
- Drop rows with missing values
- One-hot encode Industry
- Create X (features) and y (target)
- Print X shape
"""

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


CSV_PATH = "Startup Growth and Funding Trends.csv"


def load_and_prepare(csv_path: str = CSV_PATH):
    # Load
    df = pd.read_csv(csv_path)

    # Select columns
    cols = [
        "Industry",
        "Funding Rounds",
        "Year Founded",
        "Employees",
        "Revenue (M USD)",
        "Valuation (M USD)",
    ]
    df = df[cols]

    # Drop rows with missing values
    df = df.dropna(how="any").reset_index(drop=True)

    # One-hot encode Industry
    df_enc = pd.get_dummies(df, columns=["Industry"], drop_first=False)

    # Split features/target
    y = df_enc["Valuation (M USD)"]
    X = df_enc.drop(columns=["Valuation (M USD)"])

    return X, y


if __name__ == "__main__":
    X, y = load_and_prepare()
    print("X shape:", X.shape)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.3f}")
    print(f"R^2: {r2:.3f}")

    # Save model and feature columns
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'startup_valuation_rf.joblib')
    cols_path = os.path.join('models', 'valuation_model_columns.joblib')
    joblib.dump(model, model_path)
    joblib.dump(list(X.columns), cols_path)
    print(f"Saved model to {model_path} and columns to {cols_path}")


