#!/usr/bin/env python3
"""
Train Valuations Model from global_startup_success_dataset
===========================================================

Train valuation model using the global startup success dataset (5000 samples).
Target: Valuation ($B) based on multiple features.
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib

warnings.filterwarnings('ignore')

DATA_PATH = Path("data/global_startup_success_dataset.csv")
OUTPUT_PATH = Path("models/valuations_model_global.pkl")


def load_data(file_path):
    """Load and explore data"""
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
    
    print("\nDataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {len(df.columns) - 1}")
    
    print("\nTarget variable statistics:")
    target_col = 'Valuation ($B)'
    print(f"  {target_col}:")
    print(f"    Mean: {df[target_col].mean():.2f} B")
    print(f"    Median: {df[target_col].median():.2f} B")
    print(f"    Min: {df[target_col].min():.2f} B")
    print(f"    Max: {df[target_col].max():.2f} B")
    print(f"    Std: {df[target_col].std():.2f} B")
    
    return df


def engineer_features(df):
    """Add engineered features"""
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 80)
    
    df = df.copy()
    
    # Create new features
    # 1. Company age
    df['Company_Age'] = 2024 - df['Founded Year']
    
    # 2. Funding per employee
    df['Funding_Per_Employee'] = df['Total Funding ($M)'] / df['Number of Employees'].replace(0, 1)
    
    # 3. Revenue per employee
    df['Revenue_Per_Employee'] = df['Annual Revenue ($M)'] / df['Number of Employees'].replace(0, 1)
    
    # 4. Revenue efficiency (revenue / funding)
    df['Revenue_Efficiency'] = df['Annual Revenue ($M)'] / df['Total Funding ($M)'].replace(0, 1)
    
    # 5. Valuation intensity (customers per billion valuation)
    df['Customer_Intensity'] = df['Customer Base (Millions)'] * 1000 / df['Valuation ($B)'].replace(0, 0.01)
    
    # 6. Social media per employee
    df['Social_Per_Employee'] = df['Social Media Followers'] / df['Number of Employees'].replace(0, 1)
    
    print("[OK] Engineered features created:")
    print("  - Company_Age")
    print("  - Funding_Per_Employee")
    print("  - Revenue_Per_Employee")
    print("  - Revenue_Efficiency")
    print("  - Customer_Intensity")
    print("  - Social_Per_Employee")
    
    return df


def prepare_features_v1(df):
    """Version 1: Basic features"""
    feature_cols = [
        'Total Funding ($M)',
        'Annual Revenue ($M)',
        'Number of Employees',
        'Industry',
        'Funding Stage'
    ]
    
    X = df[feature_cols].copy()
    y = df['Valuation ($B)'].copy()
    
    numerical = ['Total Funding ($M)', 'Annual Revenue ($M)', 'Number of Employees']
    categorical = ['Industry', 'Funding Stage']
    
    return X, y, numerical, categorical, "V1: Basic Features"


def prepare_features_v2(df):
    """Version 2: Basic + Success metrics"""
    feature_cols = [
        'Total Funding ($M)',
        'Annual Revenue ($M)',
        'Number of Employees',
        'Industry',
        'Funding Stage',
        'Success Score',
        'Customer Base (Millions)'
    ]
    
    X = df[feature_cols].copy()
    y = df['Valuation ($B)'].copy()
    
    numerical = ['Total Funding ($M)', 'Annual Revenue ($M)', 'Number of Employees', 
                 'Success Score', 'Customer Base (Millions)']
    categorical = ['Industry', 'Funding Stage']
    
    return X, y, numerical, categorical, "V2: +Success+Customers"


def prepare_features_v3(df):
    """Version 3: Basic + All engineered"""
    feature_cols = [
        'Total Funding ($M)',
        'Annual Revenue ($M)',
        'Number of Employees',
        'Industry',
        'Funding Stage',
        'Success Score',
        'Customer Base (Millions)',
        'Company_Age',
        'Funding_Per_Employee',
        'Revenue_Per_Employee',
        'Revenue_Efficiency'
    ]
    
    X = df[feature_cols].copy()
    y = df['Valuation ($B)'].copy()
    
    numerical = [col for col in feature_cols if col not in ['Industry', 'Funding Stage']]
    categorical = ['Industry', 'Funding Stage']
    
    return X, y, numerical, categorical, "V3: +All Engineered"


def prepare_features_v4(df):
    """Version 4: Everything"""
    feature_cols = [
        'Total Funding ($M)',
        'Annual Revenue ($M)',
        'Number of Employees',
        'Industry',
        'Funding Stage',
        'Success Score',
        'Customer Base (Millions)',
        'Social Media Followers',
        'Company_Age',
        'Funding_Per_Employee',
        'Revenue_Per_Employee',
        'Revenue_Efficiency',
        'Customer_Intensity',
        'Social_Per_Employee'
    ]
    
    X = df[feature_cols].copy()
    y = df['Valuation ($B)'].copy()
    
    numerical = [col for col in feature_cols if col not in ['Industry', 'Funding Stage']]
    categorical = ['Industry', 'Funding Stage']
    
    return X, y, numerical, categorical, "V4: All Features"


def create_pipeline(numerical_features, categorical_features):
    """Create preprocessing pipeline"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
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
            'model__n_estimators': [300, 400, 500],
            'model__max_depth': [20, 25, 30, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    else:  # GradientBoosting
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [200, 300, 400],
            'model__max_depth': [8, 10, 12],
            'model__learning_rate': [0.05, 0.1, 0.15],
            'model__min_samples_split': [2, 5]
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
    print("GLOBAL STARTUP VALUATIONS MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare different feature sets
    feature_configs = [
        prepare_features_v1(df),
        prepare_features_v2(df),
        prepare_features_v3(df),
        prepare_features_v4(df)
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
                
                print(f"  {model_type.upper()}: R² = {result['r2']:.4f} ({result['r2']*100:.2f}%), RMSE = {result['rmse']:.4f} B")
                
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
        print(f"   RMSE: {result['rmse']:.4f} B USD")
        print(f"   MAE: {result['mae']:.4f} B USD")
        print(f"   MAPE: {result['mape']:.2%}")
    
    # Best model
    best = all_results[0]
    print("\n" + "-" * 80)
    print("BEST MODEL")
    print("-" * 80)
    print(f"Configuration: {best['version']}")
    print(f"Model Type: {best['model_type'].upper()}")
    print(f"R² Score: {best['r2']:.4f} ({best['r2']*100:.2f}%)")
    print(f"RMSE: {best['rmse']:.4f} B USD")
    print(f"MAE: {best['mae']:.4f} B USD")
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

