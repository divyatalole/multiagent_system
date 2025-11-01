import os
import warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================
# 1. Load + Feature Engineering
# ==============================================================

def load_and_engineer(csv_path: str, log_transform: bool = False, log_features: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    df["Valuation (M USD)"] = pd.to_numeric(df["Valuation (M USD)"], errors="coerce")
    df = df.dropna(subset=["Valuation (M USD)"]).reset_index(drop=True)

    # --- Feature Engineering ---
    _funding = pd.to_numeric(df["Funding Amount (M USD)"], errors="coerce").fillna(0.0)
    _revenue = pd.to_numeric(df["Revenue (M USD)"], errors="coerce").fillna(0.0)
    _rounds = pd.to_numeric(df["Funding Rounds"], errors="coerce").fillna(0.0)
    _employees = pd.to_numeric(df["Employees"], errors="coerce").fillna(0.0)
    _mshare = pd.to_numeric(df["Market Share (%)"], errors="coerce").fillna(0.0)

    df["Funding per Round"] = _funding / _rounds.replace(0, np.nan)
    df["Revenue per Employee"] = _revenue / _employees.replace(0, np.nan)
    df["Years Since Founding"] = 2025 - df["Year Founded"]
    df["Funding_to_Revenue"] = _funding / _revenue.replace(0, np.nan)
    df["Funding_Revenue_Interaction"] = _funding * _revenue
    df["Funding_MarketShare"] = _funding * _mshare

    if log_features:
        df["Funding Amount (M USD)"] = np.log1p(_funding)
        df["Revenue (M USD)"] = np.log1p(_revenue)
    df = df.fillna(0)

    y = df["Valuation (M USD)"].values
    if log_transform:
        y = np.log1p(y)

    feature_columns = [
        "Industry",
        "Funding Rounds",
        "Funding Amount (M USD)",
        "Revenue (M USD)",
        "Employees",
        "Market Share (%)",
        "Profitable",
        "Year Founded",
        "Funding per Round",
        "Revenue per Employee",
        "Years Since Founding",
        "Funding_to_Revenue",
        "Funding_Revenue_Interaction",
        "Funding_MarketShare",
    ]
    X = df[feature_columns].copy()
    return X, y


# ==============================================================
# 2. Build Preprocessor
# ==============================================================

def build_preprocessor(categorical_features, numeric_features) -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )


# ==============================================================
# 3. Train + Ensemble
# ==============================================================

def train_and_evaluate(csv_path: str, model_out_path: str, log_transform: bool = False, use_xgb: bool = True, log_features: bool = True):
    X, y = load_and_engineer(csv_path, log_transform=log_transform, log_features=log_features)

    categorical_features = ["Industry"]
    numeric_features = [c for c in X.columns if c not in categorical_features]
    preprocessor = build_preprocessor(categorical_features, numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Base Models ---
    rf = RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42)
    gb = None
    if use_xgb:
        try:
            from xgboost import XGBRegressor  # type: ignore
            gb = XGBRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )
        except Exception:
            gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42)
    else:
        gb = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42)

    # --- Ensemble Voting Regressor ---
    ensemble = VotingRegressor([("rf", rf), ("gb", gb)])

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", ensemble)])
    pipeline.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = pipeline.predict(X_test)
    if log_transform:
        y_pred, y_test = np.expm1(y_pred), np.expm1(y_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print("\nValuation Ensemble Model Trained Successfully")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

    # --- Feature Importance (approx from RF only) ---
    pre = pipeline.named_steps["preprocess"]
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(categorical_features)
    feature_names = list(cat_names) + numeric_features
    # Access the fitted RandomForest inside the VotingRegressor
    fitted_rf: RandomForestRegressor = pipeline.named_steps["model"].estimators_[0]
    importances = fitted_rf.feature_importances_
    feature_ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Features by Importance (from RandomForest):")
    for f, imp in feature_ranking[:10]:
        print(f"- {f}: {imp:.4f}")

    # --- Example Prediction ---
    example = {
        "Industry": "FinTech",
        "Funding Rounds": 3,
        "Funding Amount (M USD)": 150,
        "Revenue (M USD)": 45,
        "Employees": 200,
        "Market Share (%)": 6.5,
        "Profitable": 1,
        "Year Founded": 2020,
        "Funding per Round": 150 / 3,
        "Revenue per Employee": 45 / 200,
        "Years Since Founding": 2025 - 2020,
        "Funding_to_Revenue": 150 / 45,
        "Funding_Revenue_Interaction": 150 * 45,
        "Funding_MarketShare": 150 * 6.5,
    }
    example_df = pd.DataFrame([example])
    example_pred = pipeline.predict(example_df)[0]
    if log_transform:
        example_pred = float(np.expm1(example_pred))
    print(f"\nExample Prediction (M USD): {example_pred:.2f}")

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    dump(pipeline, model_out_path)


# ==============================================================
# 4. Main
# ==============================================================

if __name__ == "__main__":
    csv_path = os.path.join("data", "valuation.csv")
    model_out = os.path.join("models", "valuation_model_ensemble.joblib")
    train_and_evaluate(csv_path, model_out, log_transform=False, use_xgb=True, log_features=True)


