#!/usr/bin/env python3
"""
Train Valuations Model from global_startup_success_dataset - FAST VERSION
=========================================================================

Quick experiments without grid search for faster results.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

DATA_PATH = Path("data/global_startup_success_dataset.csv")


def load_and_engineer(df):
    """Load and engineer features"""
    df = df.copy()
    df['Company_Age'] = 2024 - df['Founded Year']
    df['Funding_Per_Employee'] = df['Total Funding ($M)'] / df['Number of Employees'].replace(0, 1)
    df['Revenue_Per_Employee'] = df['Annual Revenue ($M)'] / df['Number of Employees'].replace(0, 1)
    df['Revenue_Efficiency'] = df['Annual Revenue ($M)'] / df['Total Funding ($M)'].replace(0, 1)
    df['Customer_Intensity'] = df['Customer Base (Millions)'] * 1000 / df['Valuation ($B)'].replace(0, 0.01)
    df['Social_Per_Employee'] = df['Social Media Followers'] / df['Number of Employees'].replace(0, 1)
    return df


def prepare_features_v1(df):
    """V1: Basic"""
    feature_cols = ['Total Funding ($M)', 'Annual Revenue ($M)', 'Number of Employees', 'Industry', 'Funding Stage']
    X = df[feature_cols].copy()
    y = df['Valuation ($B)'].copy()
    numerical = ['Total Funding ($M)', 'Annual Revenue ($M)', 'Number of Employees']
    categorical = ['Industry', 'Funding Stage']
    return X, y, numerical, categorical, "V1: Basic"


def prepare_features_v2(df):
    """V2: +Success+Customers"""
    feature_cols = ['Total Funding ($M)', 'Annual Revenue ($M)', 'Number of Employees', 'Industry', 'Funding Stage',
                    'Success Score', 'Customer Base (Millions)']
    X = df[feature_cols].copy()
    y = df['Valuation ($B)'].copy()
    numerical = ['Total Funding ($M)', 'Annual Revenue ($M)', 'Number of Employees', 'Success Score', 'Customer Base (Millions)']
    categorical = ['Industry', 'Funding Stage']
    return X, y, numerical, categorical, "V2: +Success+Customers"


def prepare_features_v3(df):
    """V3: +Engineered"""
    feature_cols = ['Total Funding ($M)', 'Annual Revenue ($M)', 'Number of Employees', 'Industry', 'Funding Stage',
                    'Success Score', 'Customer Base (Millions)', 'Company_Age', 'Funding_Per_Employee', 
                    'Revenue_Per_Employee', 'Revenue_Efficiency']
    X = df[feature_cols].copy()
    y = df['Valuation ($B)'].copy()
    numerical = [col for col in feature_cols if col not in ['Industry', 'Funding Stage']]
    categorical = ['Industry', 'Funding Stage']
    return X, y, numerical, categorical, "V3: +Engineered"


def train_config(X_train, X_test, y_train, y_test, numerical, categorical, version_name, model_type):
    """Train and evaluate configuration"""
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical)
    ])
    
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=400, max_depth=25, min_samples_split=5, 
                                       min_samples_leaf=2, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=300, max_depth=10, learning_rate=0.1, random_state=42)
    
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    return {
        'version': version_name,
        'model_type': model_type,
        'pipeline': pipeline,
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred)
    }


def main():
    print("=" * 80)
    print("GLOBAL STARTUP VALUATIONS - FAST TRAINING")
    print("=" * 80)
    
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Loaded {len(df)} rows")
    
    df = load_and_engineer(df)
    
    configs = [
        prepare_features_v1(df),
        prepare_features_v2(df),
        prepare_features_v3(df)
    ]
    
    all_results = []
    
    for X, y, numerical, categorical, version_name in configs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model_type in ['rf', 'gb']:
            result = train_config(X_train, X_test, y_train, y_test, numerical, categorical, version_name, model_type)
            all_results.append(result)
            print(f"{version_name} ({model_type.upper()}): R² = {result['r2']:.4f} ({result['r2']*100:.2f}%)")
    
    all_results.sort(key=lambda x: x['r2'], reverse=True)
    
    print("\n" + "=" * 80)
    print("TOP 3 MODELS")
    print("=" * 80)
    
    for i, result in enumerate(all_results[:3], 1):
        print(f"\n{i}. {result['version']} ({result['model_type'].upper()})")
        print(f"   R²: {result['r2']:.4f} ({result['r2']*100:.2f}%)")
        print(f"   RMSE: {result['rmse']:.4f} B USD")
        print(f"   MAE: {result['mae']:.4f} B USD")
        print(f"   MAPE: {result['mape']:.2%}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    results = main()
