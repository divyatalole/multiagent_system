# Funding Trends Forecast with Prophet
import os
import pandas as pd
from datetime import datetime

# Load and aggregate into monthly series compatible with Prophet
CSV_PATH = "big_startup_secsees_dataset.csv"


def load_monthly_series(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse dates
    df['first_funding_at'] = pd.to_datetime(df['first_funding_at'], errors='coerce')
    # Keep necessary rows
    df = df.dropna(subset=['first_funding_at', 'funding_total_usd'])
    # Index and resample monthly sums
    df = df.set_index('first_funding_at')
    monthly = df['funding_total_usd'].resample('M').sum().reset_index()
    # Prophet format
    ts = monthly.rename(columns={'first_funding_at': 'ds', 'funding_total_usd': 'y'})
    print(ts.head())
    return ts


if __name__ == "__main__":
    ts = load_monthly_series()

    # Train Prophet model
    try:
        from prophet import Prophet
    except Exception:  # fallback for older install name
        from fbprophet import Prophet  # type: ignore

    model = Prophet()
    model.fit(ts)
    print("Prophet model trained.")

    # Forecast next 12 months
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)
    cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    print(forecast[cols].tail(12))

    # Save model
    import joblib
    os.makedirs('models', exist_ok=True)
    out_path = os.path.join('models', 'market_trend_prophet.joblib')
    joblib.dump(model, out_path)
    print(f"Saved Prophet model to {out_path}")
