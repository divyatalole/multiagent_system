import os
import warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================
# 1. Load + Feature Engineering
# ==============================================================

def load_and_engineer(csv_path: str, log_transform: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    df["Valuation (M USD)"] = pd.to_numeric(df["Valuation (M USD)"], errors="coerce")
    df = df.dropna(subset=["Valuation (M USD)"]).reset_index(drop=True)

    # --- Feature Engineering ---
    df["Funding per Round"] = df["Funding Amount (M USD)"] / df["Funding Rounds"].replace(0, np.nan)
    df["Revenue per Employee"] = df["Revenue (M USD)"] / df["Employees"].replace(0, np.nan)
    df["Years Since Founding"] = 2025 - df["Year Founded"]
    df["Funding_to_Revenue"] = df["Funding Amount (M USD)"] / df["Revenue (M USD)"].replace(0, np.nan)
    df = df.fillna(0)

    # Optional log transform on target
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
    ]
    X = df[feature_columns].copy()
    return X, y


# ==============================================================
# 2. Pipeline Builder
# ==============================================================

def build_pipeline(categorical_features, numeric_features, use_gb: bool = False) -> Pipeline:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    model = GradientBoostingRegressor(random_state=42) if use_gb else RandomForestRegressor(random_state=42)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipeline


# ==============================================================
# 3. Training + Evaluation
# ==============================================================

def train_and_evaluate(csv_path: str, model_out_path: str, use_gb: bool = False, log_transform: bool = False):
    X, y = load_and_engineer(csv_path, log_transform=log_transform)

    categorical_features = ["Industry"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    pipeline = build_pipeline(categorical_features, numeric_features, use_gb)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Hyperparameter tuning ---
    param_grid = (
        {
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }
        if not use_gb
        else {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        }
    )

    search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="r2", verbose=1)
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    # --- Evaluation ---
    y_pred = best_pipeline.predict(X_test)
    if log_transform:
        y_pred, y_test = np.expm1(y_pred), np.expm1(y_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print("\nValuation Model Trained Successfully")
    print(f"Best Model Params: {search.best_params_}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

    # --- Feature Importances ---
    preprocessor = best_pipeline.named_steps["preprocess"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    feature_names = list(cat_names) + numeric_features
    importances = best_pipeline.named_steps["model"].feature_importances_
    feature_ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Features by Importance:")
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
    }
    example_df = pd.DataFrame([example])
    example_pred = best_pipeline.predict(example_df)[0]
    if log_transform:
        example_pred = float(np.expm1(example_pred))
    print(f"\nExample Prediction (M USD): {example_pred:.2f}")

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    dump(best_pipeline, model_out_path)


# ==============================================================
# 4. Main
# ==============================================================

if __name__ == "__main__":
    csv_path = os.path.join("data", "valuation.csv")
    model_out = os.path.join("models", "valuation_model_optimized.joblib")
    # Try Gradient Boosting + log-transform for best results
    train_and_evaluate(csv_path, model_out, use_gb=True, log_transform=True)


