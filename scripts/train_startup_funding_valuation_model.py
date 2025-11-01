#!/usr/bin/env python3
"""
Train Startup Funding Valuation Model
=====================================

Complete training script following the specified requirements:
1. Load and clean data from startup_funding.csv
2. Feature engineering and preprocessing
3. Model training and evaluation
4. Save final model
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# ML imports
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path("data/startup_funding.csv")
OUTPUT_PATH = Path("models/startup_funding_valuation_model.pkl")


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
    print("STEP 2: CLEANING TARGET VARIABLE (Amount in USD)")
    print("=" * 80)
    
    # Create a copy
    df = df.copy()
    
    # Convert Amount in USD to string for cleaning
    df['Amount in USD'] = df['Amount in USD'].astype(str)
    
    # Remove non-numeric characters (commas, currency symbols)
    df['Amount in USD'] = df['Amount in USD'].str.replace(',', '', regex=False)
    df['Amount in USD'] = df['Amount in USD'].str.replace('₹', '', regex=False)
    df['Amount in USD'] = df['Amount in USD'].str.replace('$', '', regex=False)
    df['Amount in USD'] = df['Amount in USD'].str.replace('USD', '', regex=False)
    df['Amount in USD'] = df['Amount in USD'].str.replace(' ', '', regex=False)
    
    # Handle special values
    special_values = ['undisclosed', 'unknown', 'nan', '']
    df['Amount in USD'] = df['Amount in USD'].str.lower()
    
    # Replace special values with NaN
    for val in special_values:
        df.loc[df['Amount in USD'] == val, 'Amount in USD'] = np.nan
    
    # Convert to numeric
    df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')
    
    initial_count = len(df)
    
    # CRUCIALLY: Drop rows with missing or zero amounts
    df = df.dropna(subset=['Amount in USD'])
    df = df[df['Amount in USD'] > 0]
    
    dropped = initial_count - len(df)
    print(f"[OK] Cleaned target variable")
    print(f"[OK] Dropped {dropped} rows with missing/zero/invalid amounts")
    print(f"[OK] Remaining: {len(df)} rows")
    
    return df


def normalize_categories(text):
    """Normalize category text"""
    if pd.isna(text):
        return "Unknown"
    
    text = str(text).strip()
    
    # Common normalizations
    normalizations = {
        # Cities
        'bangalore': 'Bengaluru',
        'gurgaon': 'Gurugram',
        'delhi': 'New Delhi',
        'noida': 'Noida',
        
        # Industries
        'e-commerce': 'E-Commerce',
        'ecommerce': 'E-Commerce',
        'ecommerce': 'E-Commerce',
        'fintech': 'FinTech',
        'edtech': 'EdTech',
        'healthtech': 'HealthTech',
        'saas': 'SaaS',
        
        # Investment Types
        'seed round': 'Seed Funding',
        'seed funding': 'Seed Funding',
        'series a': 'Series A',
        'series b': 'Series B',
        'series c': 'Series C',
        'series d': 'Series D',
    }
    
    text_lower = text.lower()
    for key, value in normalizations.items():
        if text_lower == key:
            return value
    
    return text  # Return original if no normalization


def consolidate_rare_categories(series, top_n=25, category_name=""):
    """
    Consolidate rare categories into 'Other'
    
    Args:
        series: Pandas Series with categories
        top_n: Number of top categories to keep
        category_name: Name for logging
    
    Returns:
        Series with consolidated categories
    """
    # Count frequencies
    counts = series.value_counts()
    
    # Get top N categories
    top_categories = counts.head(top_n).index.tolist()
    
    # Replace rare categories with 'Other'
    series = series.copy()
    series[~series.isin(top_categories)] = 'Other'
    
    num_kept = len(top_categories)
    num_consolidated = len(counts) - num_kept
    
    print(f"  {category_name}: Kept {num_kept} top categories, consolidated {num_consolidated} into 'Other'")
    
    return series


def clean_feature_variables(df):
    """Clean and consolidate feature variables"""
    print("\n" + "=" * 80)
    print("STEP 3: CLEANING FEATURE VARIABLES")
    print("=" * 80)
    
    df = df.copy()
    
    # Select only the columns we need
    feature_cols = ['Industry Vertical', 'City  Location', 'InvestmentnType']
    
    # Rename for consistency
    df = df.rename(columns={
        'Industry Vertical': 'IndustryVertical',
        'City  Location': 'CityLocation',
        'InvestmentnType': 'InvestmentType',
        'Amount in USD': 'AmountInUSD'
    })
    
    print("[OK] Normalizing categories...")
    
    # Normalize each feature column
    df['IndustryVertical'] = df['IndustryVertical'].apply(normalize_categories)
    df['CityLocation'] = df['CityLocation'].apply(normalize_categories)
    df['InvestmentType'] = df['InvestmentType'].apply(normalize_categories)
    
    print("[OK] Consolidating rare categories...")
    
    # Consolidate rare categories
    df['IndustryVertical'] = consolidate_rare_categories(
        df['IndustryVertical'], top_n=25, category_name="Industry"
    )
    df['CityLocation'] = consolidate_rare_categories(
        df['CityLocation'], top_n=25, category_name="City"
    )
    df['InvestmentType'] = consolidate_rare_categories(
        df['InvestmentType'], top_n=30, category_name="Investment"
    )
    
    # Handle any remaining missing values
    for col in feature_cols:
        if col == 'Industry Vertical':
            new_col = 'IndustryVertical'
        elif col == 'City  Location':
            new_col = 'CityLocation'
        else:
            new_col = 'InvestmentType'
        
        missing = df[new_col].isnull().sum()
        if missing > 0:
            df[new_col].fillna('Unknown', inplace=True)
            print(f"  Filled {missing} missing values in {new_col} with 'Unknown'")
    
    print("[OK] Feature variables cleaned")
    
    return df


def create_preprocessing_pipeline():
    """Create preprocessing pipeline with OneHotEncoder"""
    print("\n" + "=" * 80)
    print("STEP 4: CREATING PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Define categorical features
    categorical_features = ['IndustryVertical', 'CityLocation', 'InvestmentType']
    
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'),
             categorical_features)
        ],
        remainder='passthrough'
    )
    
    print("[OK] Preprocessing pipeline created")
    print(f"  Categorical features: {categorical_features}")
    print("  Encoder: OneHotEncoder with handle_unknown='ignore'")
    
    return preprocessor


def train_and_evaluate(df):
    """Train model and evaluate performance"""
    print("\n" + "=" * 80)
    print("STEP 5: MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    # Extract features and target
    feature_cols = ['IndustryVertical', 'CityLocation', 'InvestmentType']
    X = df[feature_cols]
    y = df['AmountInUSD'].copy()
    
    # Apply log-transform to target
    y_log = np.log1p(y)
    print("[OK] Applied log-transform to target variable")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Split data
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42, stratify=None
    )
    
    # Also keep original test target for evaluation
    _, _, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"[OK] Split data: {len(X_train)} train, {len(X_test)} test")
    
    # Select model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("[OK] Using RandomForestRegressor")
    
    # Build full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train pipeline
    print("\nTraining model...")
    pipeline.fit(X_train, y_train_log)
    print("[OK] Model trained successfully")
    
    # Make predictions
    print("\nEvaluating on test set...")
    y_pred_log = pipeline.predict(X_test)
    
    # Reverse log-transform
    y_pred = np.expm1(y_pred_log)
    print("[OK] Reversed log-transform on predictions")
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "-" * 80)
    print("MODEL EVALUATION RESULTS")
    print("-" * 80)
    print(f"R-squared (R²):     {r2:.4f} ({r2*100:.2f}%)")
    print(f"RMSE:               ${rmse:,.2f}")
    print(f"MAE:                ${mae:,.2f}")
    
    # Additional analysis
    print("\n" + "-" * 80)
    print("ADDITIONAL ANALYSIS")
    print("-" * 80)
    print(f"Test set size:      {len(y_test)}")
    print(f"Mean actual value:  ${y_test.mean():,.2f}")
    print(f"Mean prediction:    ${y_pred.mean():,.2f}")
    
    return pipeline, X_test, y_test, y_pred


def save_final_model(df, pipeline):
    """Retrain on all data and save final model"""
    print("\n" + "=" * 80)
    print("STEP 6: SAVING FINAL MODEL")
    print("=" * 80)
    
    # Extract features and target
    feature_cols = ['IndustryVertical', 'CityLocation', 'InvestmentType']
    X = df[feature_cols]
    y = df['AmountInUSD']
    
    # Apply log-transform
    y_log = np.log1p(y)
    
    # Retrain on ALL data
    print("Retraining on 100% of data...")
    pipeline.fit(X, y_log)
    print("[OK] Retrained on all data")
    
    # Save to file
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"[OK] Model saved to: {OUTPUT_PATH}")
    
    # Also save metadata
    metadata = {
        'feature_columns': feature_cols,
        'model_type': 'RandomForestRegressor',
        'training_samples': len(df),
        'n_estimators': 200,
        'target_transform': 'log1p'
    }
    
    import json
    metadata_path = OUTPUT_PATH.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Metadata saved to: {metadata_path}")


def main():
    """Main execution"""
    try:
        # Step 1: Load data
        df = load_data(DATA_PATH)
        
        # Step 2: Clean target variable
        df = clean_target_variable(df)
        
        # Step 3: Clean feature variables
        df = clean_feature_variables(df)
        
        print("\n" + "-" * 80)
        print("DATA CLEANING SUMMARY")
        print("-" * 80)
        print(f"Final dataset: {len(df)} rows")
        print(f"Features: IndustryVertical, CityLocation, InvestmentType")
        print(f"Target: AmountInUSD")
        print(f"Mean funding: ${df['AmountInUSD'].mean():,.2f}")
        print(f"Median funding: ${df['AmountInUSD'].median():,.2f}")
        
        # Step 5: Train and evaluate
        pipeline, X_test, y_test, y_pred = train_and_evaluate(df)
        
        # Step 6: Save final model
        save_final_model(df, pipeline)
        
        # Final summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"[OK] Model successfully trained and saved")
        print(f"[OK] Training samples: {len(df)}")
        print(f"[OK] Test performance: R^2 = {r2_score(y_test, y_pred):.4f}")
        print(f"\nModel location: {OUTPUT_PATH}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

