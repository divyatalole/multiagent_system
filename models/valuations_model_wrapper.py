"""
Valuations Model Wrapper
========================

Wrapper class to use the trained valuations model for predicting startup valuations.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional


class ValuationsModel:
    """
    Wrapper for the valuations model
    
    This model predicts startup valuation based on:
    - Funding Amount (M USD)
    - Revenue (M USD)
    - Employees
    - Industry (categorical)
    - Funding Rounds
    
    Usage:
        model = ValuationsModel()
        prediction = model.predict(
            funding_amount=100.0,
            revenue=50.0,
            employees=500,
            industry="FinTech",
            funding_rounds=3
        )
    """
    
    def __init__(self, model_path: str = "models/valuations_model.pkl"):
        """
        Initialize the model
        
        Args:
            model_path: Path to the trained model pickle file
        """
        self.model_path = Path(model_path)
        self.model = None
        self.loaded = False
        
        # Try to load the model
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.loaded = True
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}: {e}")
        else:
            print(f"Warning: Model file not found at {model_path}")
    
    def load(self) -> bool:
        """
        Load the model
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if self.loaded:
            return True
        
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.loaded = True
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        
        return False
    
    def predict(
        self,
        funding_amount: float,
        revenue: float,
        employees: int,
        industry: str,
        funding_rounds: int
    ) -> Dict[str, Any]:
        """
        Predict valuation for a startup
        
        Args:
            funding_amount: Funding amount in millions USD
            revenue: Revenue in millions USD
            employees: Number of employees
            industry: Industry category (e.g., "FinTech", "AI", "HealthTech")
            funding_rounds: Number of funding rounds
            
        Returns:
            Dictionary with prediction and metadata
        """
        if not self.loaded:
            if not self.load():
                return {
                    'prediction': 0.0,
                    'error': 'Model not loaded',
                    'success': False
                }
        
        try:
            # Prepare input as DataFrame
            input_data = pd.DataFrame({
                'Funding Amount (M USD)': [funding_amount],
                'Revenue (M USD)': [revenue],
                'Employees': [employees],
                'Industry': [industry],
                'Funding Rounds': [funding_rounds]
            })
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            return {
                'prediction': float(prediction),
                'prediction_m_usd': float(prediction),
                'success': True,
                'input': {
                    'funding_amount': funding_amount,
                    'revenue': revenue,
                    'employees': employees,
                    'industry': industry,
                    'funding_rounds': funding_rounds
                }
            }
            
        except Exception as e:
            return {
                'prediction': 0.0,
                'error': str(e),
                'success': False
            }
    
    def predict_batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict valuations for multiple startups
        
        Args:
            inputs: List of dictionaries with startup data
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for input_data in inputs:
            result = self.predict(
                funding_amount=input_data.get('funding_amount', 0.0),
                revenue=input_data.get('revenue', 0.0),
                employees=input_data.get('employees', 0),
                industry=input_data.get('industry', 'Unknown'),
                funding_rounds=input_data.get('funding_rounds', 1)
            )
            results.append(result)
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.loaded


# Make it importable
__all__ = ['ValuationsModel']

