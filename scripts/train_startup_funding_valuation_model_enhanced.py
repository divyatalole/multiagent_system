#!/usr/bin/env python3
"""
Train Enhanced Startup Funding Valuation Model
==============================================

Enhanced version with additional features for better performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

warnings.filterwarnings('ignore')

DATA_PATH = Path("data/startup_funding.csv")
OUTPUT_PATH = Path("models/startup_funding_valuation_model_enhanced.pkl")


def load_data(file_path):
    """Load the startup funding dataset"""
    print("=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df


def clean_target_variable(df):
    """Clean the Amount in USD target variable"""
    print("\n" + "=" * 80)
    print("STEP 2: CLEANING TARGET VARIABLE")
    print("=" * 80)
    
    df = df.copy()
    df['Amount in USD'] = df['Amount in USD'].astype(str)
    
    # Remove non-numeric characters
    df['Amount in USD'] = df['Amount in USD'].str.replace(',', '', regex=False)
    df['Amount in USD'] = df['Amount in USD'].str.replace('₹', '', regex=False)
    df['Amount in USD'] = df['Amount in USD'].str.replace('$', '', regex=False)
    df['Amount in USD'] = df['Amount in USD'].str.replace('USD', '', regex=False)
    df['Amount in USD'] = df['Amount in USD'].str.replace(' ', '', regex=False)
    
    # Handle special values
    special_values = ['undisclosed', 'unknown', 'nan', '']
    df['Amount in USD'] = df['Amount in USD'].str.lower()
    for val in special_values:
        df.loc[df['Amount in USD'] == val, 'Amount in USD'] = np.nan
    
    # Convert to numeric
    df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')
    
    initial_count = len(df)
    df = df.dropna(subset=['Amount in USD'])
    df = df[df['Amount in USD'] > 0]
    
    dropped = initial_count - len(df)
    print(f"[OK] Dropped {dropped} rows with missing/zero/invalid amounts")
    print(f"[OK] Remaining: {len(df)} rows")
    
    return df


def normalize_categories(text):
    """Normalize category text"""
    if pd.isna(text):
        return "Unknown"
    
    text = str(text).strip()
    normalizations = {
        'bangalore': 'Bengaluru', 'gurgaon': 'Gurugram', 'delhi': 'New Delhi',
        'noida': 'Noida', 'e-commerce': 'E-Commerce', 'ecommerce': 'E-Commerce',
        'fintech': 'FinTech', 'edtech': 'EdTech', 'healthtech': 'HealthTech',
        'saas': 'SaaS', 'seed round': 'Seed Funding', 'seed funding': 'Seed Funding',
        'series a': 'Series A', 'series b': 'Series B', 'series c': 'Series C',
        'series d': 'Series D'
    }
    
    text_lower = text.lower()
    for key, value in normalizations.items():
        if text_lower == key:
            return value
    
    return text


def consolidate_rare_categories(series, top_n=30, category_name=""):
    """Consolidate rare categories into 'Other'"""
    counts = series.value_counts()
    top_categories = counts.head(top_n).index.tolist()
    series = series.copy()
    series[~series.isin(top_categories)] = 'Other'
    
    num_kept = len(top_categories)
    num_consolidated = len(counts) - num_kept
    print(f"  {category_name}: Kept {num_kept} top categories, consolidated {num_consolidated} into 'Other'")
    
    return series


def clean_and_add_features(df):
    """Clean features and add additional features"""
    print("\n" + "=" * 80)
    print("STEP 3: CLEANING AND ENGINEERING FEATURES")
    print("=" * 80)
    
    df = df.copy()
    
    # Rename columns
    df = df.rename(columns={
        'Industry Vertical': 'IndustryVertical',
        'City  Location': 'CityLocation',
        'InvestmentnType': 'InvestmentType',
        'Amount in USD': 'AmountInUSD',
        'SubVertical': 'SubVertical',
        'Date dd/mm/yyyy': 'Date'
    })
    
    print("[OK] Normalizing categories...")
    df['IndustryVertical'] = df['IndustryVertical'].apply(normalize_categories)
    df['CityLocation'] = df['CityLocation'].apply(normalize_categories)
    df['InvestmentType'] = df['InvestmentType'].apply(normalize_categories)
    df['SubVertical'] = df['SubVertical'].apply(normalize_categories)
    
    print("[OK] Consolidating rare categories...")
    df['IndustryVertical'] = consolidate_rare_categories(df['IndustryVertical'], top_n=30, category_name="Industry")
    df['CityLocation'] = consolidate_rare_categories(df['CityLocation'], top_n=30, category_name="City")
    df['InvestmentType'] = consolidate_rare_categories(df['InvestmentType'], top_n=30, category_name="Investment")
    df['SubVertical'] = consolidate_rare_categories(df['SubVertical'], top_n=25, category_name="SubVertical")
    
    # Handle missing values
    for col in ['IndustryVertical', 'CityLocation', 'InvestmentType', 'SubVertical']:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col].fillna('Unknown', inplace=True)
            print(f"  Filled {missing} missing values in {col}")
    
    print("[OK] Features cleaned")
    
    return df


def create_pipeline():
    """Create preprocessing pipeline"""
    print("\n" + "=" * 80)
    print("STEP 4: CREATING PREPROCESSING PIPELINE")
    print("=" * 80)
    
    categorical_features = ['IndustryVertical', 'CityLocation', 'InvestmentType', 'SubVertical']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'),
             categorical_features)
        ],
        remainder='passthrough'
    )
    
    print("[OK] Preprocessing pipeline created")
    print(f"  Categorical features: {categorical_features}")
    
    return preprocessor


def train_and_evaluate(df):
    """Train model and evaluate"""
    print("\n" + "=" * 80)
    print("STEP 5: MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    # Extract features
    feature_cols = ['IndustryVertical', 'CityLocation', 'InvestmentType', 'SubVertical']
    X = df[feature_cols]
    y = df['AmountInUSD'].copy()
    
    # Log-transform
    y_log = np.log1p(y)
    print("[OK] Applied log-transform to target")
    
    # Preprocessing
    preprocessor = create_pipeline()
    
    # Split
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"[OK] Split: {len(X_train)} train, {len(X_test)} test")
    
    # Model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    print("\nTraining model...")
    pipeline.fit(X_train, y_train_log)
    print("[OK] Model trained")
    
    # Predict
    print("\nEvaluating...")
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    print("[OK] Predictions made")
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "-" * 80)
    print("MODEL EVALUATION RESULTS")
    print("-" * 80)
    print(f"R-squared (R^2):     {r2:.4f} ({r2*100:.2f}%)")
    print(f"RMSE:               ${rmse:,.2f}")
    print(f"MAE:                ${mae:,.2f}")
    print(f"Test size:          {len(y_test)}")
    print(f"Mean actual:        ${y_test.mean():,.2f}")
    print(f"Mean predicted:     ${y_pred.mean():,.2f}")
    
    return pipeline, X_test, y_test, y_pred


def save_model(df, pipeline):
    """Save final model"""
    print("\n" + "=" * 80)
    print("STEP 6: SAVING FINAL MODEL")
    print("=" * 80)
    
    feature_cols = ['IndustryVertical', 'CityLocation', 'InvestmentType', 'SubVertical']
    X = df[feature_cols]
    y = df['AmountInUSD']
    y_log = np.log1p(y)
    
    print("Retraining on all data...")
    pipeline.fit(X, y_log)
    print("[OK] Retrained on all data")
    
    # Save
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"[OK] Model saved to: {OUTPUT_PATH}")
    
    # Metadata
    metadata = {
        'feature_columns': feature_cols,
        'model_type': 'RandomForestRegressor',
        'training_samples': len(df),
        'n_estimators': 300,
        'target_transform': 'log1p'
    }
    
    import json
    metadata_path = OUTPUT_PATH.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved")


def main():
    try:
        df = load_data(DATA_PATH)
        df = clean_target_variable(df)
        df = clean_and_add_features(df)
        
        print("\n" + "-" * 80)
        print("DATA CLEANING SUMMARY")
        print("-" * 80)
        print(f"Final dataset: {len(df)} rows")
        print(f"Features: IndustryVertical, CityLocation, InvestmentType, SubVertical")
        print(f"Mean funding: ${df['AmountInUSD'].mean():,.2f}")
        
        pipeline, X_test, y_test, y_pred = train_and_evaluate(df)
        save_model(df, pipeline)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"[OK] Model trained and saved")
        print(f"[OK] Training samples: {len(df)}")
        print(f"[OK] Test R^2: {r2_score(y_test, y_pred):.4f}")
        print(f"\nModel: {OUTPUT_PATH}")
        print("=" * 80)
        
        return 0
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

