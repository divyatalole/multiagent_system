import os
import warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def load_and_engineer(csv_path: str, log_features: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    df["Valuation (M USD)"] = pd.to_numeric(df["Valuation (M USD)"], errors="coerce")
    df = df.dropna(subset=["Valuation (M USD)"]).reset_index(drop=True)

    _funding = pd.to_numeric(df["Funding Amount (M USD)"], errors="coerce").fillna(0.0)
    _revenue = pd.to_numeric(df["Revenue (M USD)"], errors="coerce").fillna(0.0)
    _rounds = pd.to_numeric(df["Funding Rounds"], errors="coerce").fillna(0.0)
    _employees = pd.to_numeric(df["Employees"], errors="coerce").fillna(0.0)
    _mshare = pd.to_numeric(df["Market Share (%)"], errors="coerce").fillna(0.0)

    # Engineered features from originals
    df["Funding per Round"] = _funding / _rounds.replace(0, np.nan)
    df["Revenue per Employee"] = _revenue / _employees.replace(0, np.nan)
    df["Years Since Founding"] = 2025 - df["Year Founded"]
    df["Funding_to_Revenue"] = _funding / _revenue.replace(0, np.nan)
    df["Funding_Revenue_Interaction"] = _funding * _revenue
    df["Funding_MarketShare"] = _funding * _mshare

    # Selective log for skewed base features
    if log_features:
        df["Funding Amount (M USD)"] = np.log1p(_funding)
        df["Revenue (M USD)"] = np.log1p(_revenue)

    df = df.fillna(0)

    y = df["Valuation (M USD)"].values
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


def build_preprocessor(categorical_features, numeric_features) -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )


def train_and_evaluate(csv_path: str, model_out_path: str):
    X, y = load_and_engineer(csv_path, log_features=True)

    categorical_features = ["Industry"]
    numeric_features = [c for c in X.columns if c not in categorical_features]
    preprocessor = build_preprocessor(categorical_features, numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Base estimators
    rf = RandomForestRegressor(random_state=42)
    # Try to use XGBoost; fall back to GradientBoosting
    try:
        from xgboost import XGBRegressor  # type: ignore

        xgb = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        estimators = [("rf", rf), ("xgb", xgb)]
        weights_key = "model__weights"
        # Tuning grid with XGB
        param_grid = {
            "model__rf__n_estimators": [400, 800],
            "model__rf__max_depth": [None, 20],
            "model__rf__min_samples_leaf": [1, 2],
            "model__xgb__n_estimators": [400, 800],
            "model__xgb__max_depth": [4, 6],
            "model__xgb__learning_rate": [0.03, 0.05],
            "model__xgb__subsample": [0.8, 1.0],
            "model__xgb__colsample_bytree": [0.8, 1.0],
            weights_key: [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)],
        }
    except Exception:
        gb = GradientBoostingRegressor(random_state=42)
        estimators = [("rf", rf), ("gb", gb)]
        weights_key = "model__weights"
        # Tuning grid without XGB
        param_grid = {
            "model__rf__n_estimators": [400, 800],
            "model__rf__max_depth": [None, 20],
            "model__rf__min_samples_leaf": [1, 2],
            "model__gb__n_estimators": [300, 600],
            "model__gb__learning_rate": [0.05, 0.1],
            "model__gb__max_depth": [3, 4],
            weights_key: [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)],
        }

    ensemble = VotingRegressor(estimators=estimators)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", ensemble)])

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print("\nValuation Tuned Ensemble Model Trained Successfully")
    print(f"Best Params: {search.best_params_}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

    # Feature importances from RF component
    pre = best_pipeline.named_steps["preprocess"]
    cat_names = pre.named_transformers_["cat"].get_feature_names_out(categorical_features)
    feature_names = list(cat_names) + numeric_features
    # Pull fitted RF from VotingRegressor
    vr = best_pipeline.named_steps["model"]
    fitted_rf = None
    for name, est in zip(vr.estimators, vr.estimators_):
        if name == "rf":
            fitted_rf = est
            break
    if fitted_rf is not None and hasattr(fitted_rf, "feature_importances_"):
        importances = fitted_rf.feature_importances_
        ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        print("\nTop 10 Features by Importance (from RandomForest):")
        for f, imp in ranking[:10]:
            print(f"- {f}: {imp:.4f}")

    # Example prediction
    example = {
        "Industry": "FinTech",
        "Funding Rounds": 3,
        "Funding Amount (M USD)": np.log1p(150.0),  # match log-features
        "Revenue (M USD)": np.log1p(45.0),          # match log-features
        "Employees": 200,
        "Market Share (%)": 6.5,
        "Profitable": 1,
        "Year Founded": 2020,
        "Funding per Round": 150.0 / 3.0,
        "Revenue per Employee": 45.0 / 200.0,
        "Years Since Founding": 2025 - 2020,
        "Funding_to_Revenue": 150.0 / 45.0,
        "Funding_Revenue_Interaction": 150.0 * 45.0,
        "Funding_MarketShare": 150.0 * 6.5,
    }
    example_df = pd.DataFrame([example])
    example_pred = best_pipeline.predict(example_df)[0]
    print(f"\nExample Prediction (M USD): {example_pred:.2f}")

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    dump(best_pipeline, model_out_path)


if __name__ == "__main__":
    csv_path = os.path.join("data", "valuation.csv")
    model_out = os.path.join("models", "valuation_model_ensemble_tuned.joblib")
    train_and_evaluate(csv_path, model_out)



