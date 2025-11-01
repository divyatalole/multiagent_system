import os
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure target is numeric and drop invalid rows
    df["Valuation (M USD)"] = pd.to_numeric(df["Valuation (M USD)"], errors="coerce")
    df = df.dropna(subset=["Valuation (M USD)"])
    return df


def build_pipeline(categorical_features, numeric_features) -> Pipeline:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return pipeline


def get_feature_names(preprocessor: ColumnTransformer, categorical_features, numeric_features):
    # Retrieve names post OneHotEncoder; numeric pass-through keeps original names
    cat_encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features)
    return list(cat_names) + list(numeric_features)


def train_and_evaluate(csv_path: str, model_out_path: str):
    warnings.filterwarnings("ignore", category=UserWarning)

    df = load_and_clean(csv_path)

    feature_columns = [
        "Industry",
        "Funding Rounds",
        "Funding Amount (M USD)",
        "Revenue (M USD)",
        "Employees",
        "Market Share (%)",
        "Profitable",
        "Year Founded",
    ]
    target_column = "Valuation (M USD)"

    categorical_features = ["Industry", "Region", "Exit Status"]
    # Only those listed in X that are numeric; Region/Exit Status are not in X per spec
    # We'll still use Region and Exit Status in preprocessing if present in data X? Spec excludes them from X.
    # Follow spec strictly for X; include only defined columns.
    numeric_features = [
        "Funding Rounds",
        "Funding Amount (M USD)",
        "Revenue (M USD)",
        "Employees",
        "Market Share (%)",
        "Profitable",
        "Year Founded",
    ]

    # Build X strictly per spec; categorical only includes Industry from X
    X = df[feature_columns].copy()
    y = df[target_column].values

    # Preprocessor should only reference columns present in X
    categorical_in_X = ["Industry"]
    pipeline = build_pipeline(categorical_in_X, numeric_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    # Compute RMSE compatible with older sklearn versions
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    # Feature importances mapped back to names
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    feature_names = get_feature_names(preprocessor, categorical_in_X, numeric_features)
    importances = pipeline.named_steps["model"].feature_importances_
    # Guard against mismatch lengths
    k = min(len(importances), len(feature_names))
    feature_ranking = sorted(
        zip(feature_names[:k], importances[:k]), key=lambda x: x[1], reverse=True
    )

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    dump(pipeline, model_out_path)

    # Output per spec
    print("Valuation model trained successfully.")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("Top 5 Features by Importance:")
    for name, score in feature_ranking[:5]:
        print(f"- {name}: {score:.4f}")

    # Example prediction
    example = {
        "Industry": "FinTech",
        "Funding Rounds": 3,
        "Funding Amount (M USD)": 150,
        "Revenue (M USD)": 45,
        "Employees": 200,
        "Market Share (%)": 6.5,
        "Profitable": 1,
        "Year Founded": 2020,
    }
    example_df = pd.DataFrame([example])
    pred = pipeline.predict(example_df)[0]
    print(f"Example Prediction (M USD): {pred:.2f}")


if __name__ == "__main__":
    csv_path = os.path.join("data", "valuation.csv")
    model_out = os.path.join("models", "valuation_model.joblib")
    train_and_evaluate(csv_path, model_out)


