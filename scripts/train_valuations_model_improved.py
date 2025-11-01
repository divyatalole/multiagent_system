#!/usr/bin/env python3
"""
Train Improved Valuations Model with Feature Engineering
========================================================

Experiment with more features and hyperparameters to improve R² beyond 64.89%.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib

warnings.filterwarnings('ignore')

DATA_PATH = Path("data/startup_data.csv")


def load_data(file_path):
    """Load and explore data"""
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def engineer_features(df):
    """Add engineered features"""
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 80)
    
    df = df.copy()
    
    # Create new features
    # 1. Funding per employee
    df['Funding_Per_Employee'] = df['Funding Amount (M USD)'] / df['Employees'].replace(0, 1)
    
    # 2. Revenue per employee
    df['Revenue_Per_Employee'] = df['Revenue (M USD)'] / df['Employees'].replace(0, 1)
    
    # 3. Funding efficiency (revenue / funding)
    df['Funding_Efficiency'] = df['Revenue (M USD)'] / df['Funding Amount (M USD)'].replace(0, 1)
    
    # 4. Average funding per round
    df['Avg_Funding_Per_Round'] = df['Funding Amount (M USD)'] / df['Funding Rounds'].replace(0, 1)
    
    # 5. Company age (from 2024)
    df['Company_Age'] = 2024 - df['Year Founded']
    
    # 6. Market share * revenue (market size indicator)
    df['Market_Impact'] = df['Market Share (%)'] * df['Revenue (M USD)']
    
    print("[OK] Engineered features created:")
    print("  - Funding_Per_Employee")
    print("  - Revenue_Per_Employee")
    print("  - Funding_Efficiency")
    print("  - Avg_Funding_Per_Round")
    print("  - Company_Age")
    print("  - Market_Impact")
    
    return df


def prepare_features_v1(df):
    """Version 1: Original features only"""
    feature_cols = [
        'Funding Amount (M USD)',
        'Revenue (M USD)',
        'Employees',
        'Industry',
        'Funding Rounds'
    ]
    
    X = df[feature_cols].copy()
    y = df['Valuation (M USD)'].copy()
    
    numerical = ['Funding Amount (M USD)', 'Revenue (M USD)', 'Employees', 'Funding Rounds']
    categorical = ['Industry']
    
    return X, y, numerical, categorical, "V1: Original"


def prepare_features_v2(df):
    """Version 2: Original + Company Age + Profitable + Market Share"""
    feature_cols = [
        'Funding Amount (M USD)',
        'Revenue (M USD)',
        'Employees',
        'Industry',
        'Funding Rounds',
        'Profitable',
        'Market Share (%)',
        'Company_Age'
    ]
    
    X = df[feature_cols].copy()
    y = df['Valuation (M USD)'].copy()
    
    numerical = ['Funding Amount (M USD)', 'Revenue (M USD)', 'Employees', 'Funding Rounds', 
                 'Market Share (%)', 'Company_Age']
    categorical = ['Industry', 'Profitable']
    
    return X, y, numerical, categorical, "V2: +Profitability+Market+Age"


def prepare_features_v3(df):
    """Version 3: Original + All Engineered Features"""
    feature_cols = [
        'Funding Amount (M USD)',
        'Revenue (M USD)',
        'Employees',
        'Industry',
        'Funding Rounds',
        'Profitable',
        'Market Share (%)',
        'Company_Age',
        'Funding_Per_Employee',
        'Revenue_Per_Employee',
        'Funding_Efficiency',
        'Avg_Funding_Per_Round',
        'Market_Impact'
    ]
    
    X = df[feature_cols].copy()
    y = df['Valuation (M USD)'].copy()
    
    numerical = [col for col in feature_cols if col != 'Industry' and col != 'Profitable']
    categorical = ['Industry', 'Profitable']
    
    return X, y, numerical, categorical, "V3: All Features"


def create_pipeline(numerical_features, categorical_features, scaler_type='standard'):
    """Create preprocessing pipeline"""
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'),
             categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def train_and_evaluate_config(X_train, X_test, y_train, y_test, numerical_features, categorical_features, 
                              model_type='rf', version_name=""):
    """Train and evaluate a model configuration"""
    
    # Create pipeline
    preprocessor = create_pipeline(numerical_features, categorical_features)
    
    # Select model
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'model__n_estimators': [200, 300, 400],
            'model__max_depth': [15, 20, 25, None],
            'model__min_samples_split': [3, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    else:  # GradientBoosting
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [200, 300],
            'model__max_depth': [8, 10, 12],
            'model__learning_rate': [0.05, 0.1, 0.15],
            'model__min_samples_split': [3, 5]
        }
    
    # Build pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Grid search
    print(f"  Grid searching {model_type.upper()}...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Evaluate
    y_pred = grid_search.best_estimator_.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    result = {
        'version': version_name,
        'model_type': model_type,
        'pipeline': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }
    
    return result


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("IMPROVED VALUATIONS MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare different feature sets
    feature_configs = [
        prepare_features_v1(df),
        prepare_features_v2(df),
        prepare_features_v3(df)
    ]
    
    # Train and evaluate all configurations
    print("\n" + "=" * 80)
    print("STEP 3: EXPERIMENTAL TRAINING")
    print("=" * 80)
    
    all_results = []
    
    for X, y, numerical, categorical, version_name in feature_configs:
        print(f"\n--- {version_name} ---")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Try both RandomForest and GradientBoosting
        for model_type in ['rf', 'gb']:
            try:
                result = train_and_evaluate_config(
                    X_train, X_test, y_train, y_test,
                    numerical, categorical, model_type, version_name
                )
                all_results.append(result)
                
                print(f"  {model_type.upper()}: R² = {result['r2']:.4f} ({result['r2']*100:.2f}%), RMSE = {result['rmse']:.2f} M USD")
                
            except Exception as e:
                print(f"  {model_type.upper()}: Error - {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort by R²
    all_results.sort(key=lambda x: x['r2'], reverse=True)
    
    print("\nTop 5 Models:")
    for i, result in enumerate(all_results[:5], 1):
        print(f"\n{i}. {result['version']} ({result['model_type'].upper()})")
        print(f"   R²: {result['r2']:.4f} ({result['r2']*100:.2f}%)")
        print(f"   RMSE: {result['rmse']:.2f} M USD")
        print(f"   MAE: {result['mae']:.2f} M USD")
        print(f"   MAPE: {result['mape']:.2%}")
    
    # Best model
    best = all_results[0]
    print("\n" + "-" * 80)
    print("BEST MODEL")
    print("-" * 80)
    print(f"Configuration: {best['version']}")
    print(f"Model Type: {best['model_type'].upper()}")
    print(f"R² Score: {best['r2']:.4f} ({best['r2']*100:.2f}%)")
    print(f"RMSE: {best['rmse']:.2f} M USD")
    print(f"MAE: {best['mae']:.2f} M USD")
    print(f"MAPE: {best['mape']:.2%}")
    print(f"\nBest Parameters:")
    for param, value in best['best_params'].items():
        print(f"  {param}: {value}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    results = main()

