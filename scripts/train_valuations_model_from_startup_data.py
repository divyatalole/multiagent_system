#!/usr/bin/env python3
"""
Train Valuations Model from startup_data.csv
============================================

Train a high-performance valuation model using numerical and categorical features.
Expected to achieve much higher R² than the categorical-only model.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# ML imports
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path("data/startup_data.csv")
OUTPUT_PATH = Path("models/valuations_model.pkl")


def load_and_explore_data(file_path):
    """Load data and print summary"""
    print("=" * 80)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
    
    print("\nDataset Summary:")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    
    print("\nTarget variable statistics:")
    print(f"  Valuation (M USD):")
    print(f"    Mean: {df['Valuation (M USD)'].mean():.2f}")
    print(f"    Median: {df['Valuation (M USD)'].median():.2f}")
    print(f"    Min: {df['Valuation (M USD)'].min():.2f}")
    print(f"    Max: {df['Valuation (M USD)'].max():.2f}")
    
    return df


def prepare_features_and_target(df):
    """Define features (X) and target (y)"""
    print("\n" + "=" * 80)
    print("STEP 2: DEFINING FEATURES AND TARGET")
    print("=" * 80)
    
    # Define features as specified
    feature_cols = [
        'Funding Amount (M USD)',  # Numerical
        'Revenue (M USD)',         # Numerical
        'Employees',               # Numerical
        'Industry',                # Categorical
        'Funding Rounds'           # Numerical (or categorical if preferred)
    ]
    
    print("[OK] Features selected:")
    for col in feature_cols:
        dtype = df[col].dtype
        print(f"  - {col} ({dtype})")
    
    # Extract X and y
    X = df[feature_cols].copy()
    y = df['Valuation (M USD)'].copy()
    
    print(f"\n[OK] Target: Valuation (M USD)")
    print(f"[OK] Feature matrix shape: {X.shape}")
    print(f"[OK] Target vector shape: {y.shape}")
    
    return X, y, feature_cols


def create_preprocessing_pipeline():
    """Create preprocessing pipeline with scaling and encoding"""
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Define numerical and categorical features
    numerical_features = ['Funding Amount (M USD)', 'Revenue (M USD)', 'Employees', 'Funding Rounds']
    categorical_features = ['Industry']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'),
             categorical_features)
        ],
        remainder='passthrough'
    )
    
    print("[OK] Preprocessing pipeline created:")
    print(f"  Numerical features: {numerical_features}")
    print(f"    -> StandardScaler")
    print(f"  Categorical features: {categorical_features}")
    print(f"    -> OneHotEncoder")
    
    return preprocessor


def train_and_evaluate_models(X, y, preprocessor):
    """Train and evaluate multiple models"""
    print("\n" + "=" * 80)
    print("STEP 4: MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"[OK] Data split: {len(X_train)} train, {len(X_test)} test")
    
    # List of models to try
    models_to_test = [
        ('RandomForest', RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        ('GradientBoosting', GradientBoostingRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        ))
    ]
    
    results = {}
    
    for model_name, model in models_to_test:
        print(f"\nTraining {model_name}...")
        
        # Build pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        print(f"[OK] {model_name} trained")
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        results[model_name] = {
            'pipeline': pipeline,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        print(f"\n{model_name} Results:")
        print(f"  R-squared: {r2:.4f} ({r2*100:.2f}%)")
        print(f"  RMSE: {rmse:.2f} M USD")
        print(f"  MAE: {mae:.2f} M USD")
        print(f"  MAPE: {mape:.2%}")
    
    # Print comparison
    print("\n" + "-" * 80)
    print("MODEL COMPARISON")
    print("-" * 80)
    for model_name, metrics in results.items():
        print(f"{model_name}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f} M USD")
    
    return results


def save_best_model(df, best_pipeline, X, y):
    """Retrain on all data and save best model"""
    print("\n" + "=" * 80)
    print("STEP 5: SAVING FINAL MODEL")
    print("=" * 80)
    
    # Retrain on ALL data
    print("Retraining best model on 100% of data...")
    best_pipeline.fit(X, y)
    print("[OK] Retrained on all data")
    
    # Save to file
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    joblib.dump(best_pipeline, OUTPUT_PATH)
    print(f"[OK] Model saved to: {OUTPUT_PATH}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'training_samples': len(df),
        'features': ['Funding Amount (M USD)', 'Revenue (M USD)', 'Employees', 'Industry', 'Funding Rounds'],
        'target': 'Valuation (M USD)',
        'preprocessing': {
            'numerical': 'StandardScaler',
            'categorical': 'OneHotEncoder'
        }
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
        df = load_and_explore_data(DATA_PATH)
        
        # Step 2: Prepare features and target
        X, y, feature_cols = prepare_features_and_target(df)
        
        # Step 3: Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline()
        
        # Step 4: Train and evaluate
        results = train_and_evaluate_models(X, y, preprocessor)
        
        # Select best model (highest R²)
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_pipeline = results[best_model_name]['pipeline']
        
        print(f"\n[OK] Best model: {best_model_name}")
        print(f"[OK] Best R²: {results[best_model_name]['r2']:.4f}")
        
        # Step 5: Save best model
        save_best_model(df, best_pipeline, X, y)
        
        # Final summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"[OK] Best model: {best_model_name}")
        print(f"[OK] R² score: {results[best_model_name]['r2']:.4f} ({results[best_model_name]['r2']*100:.2f}%)")
        print(f"[OK] RMSE: {results[best_model_name]['rmse']:.2f} M USD")
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

