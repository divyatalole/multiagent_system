"""
Startup Funding Valuation Model Wrapper
========================================

Wrapper class to use the trained valuation model
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


class StartupFundingModel:
    """
    Wrapper for the startup funding valuation model
    
    Usage:
        model = StartupFundingModel()
        prediction = model.predict(
            industry="FinTech",
            city="Bengaluru",
            investment_type="Series A",
            subvertical="Digital Lending"
        )
    """
    
    def __init__(self, model_path: str = "models/startup_funding_valuation_model.pkl"):
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
        industry: str,
        city: str,
        investment_type: str,
        subvertical: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict funding amount for a startup
        
        Args:
            industry: Industry vertical
            city: City location
            investment_type: Investment type (e.g., "Series A")
            subvertical: Optional subvertical category
            
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
            # Prepare input
            if subvertical:
                input_data = pd.DataFrame({
                    'IndustryVertical': [industry],
                    'CityLocation': [city],
                    'InvestmentType': [investment_type],
                    'SubVertical': [subvertical]
                })
            else:
                input_data = pd.DataFrame({
                    'IndustryVertical': [industry],
                    'CityLocation': [city],
                    'InvestmentType': [investment_type]
                })
            
            # Make prediction (returns log-transformed value)
            pred_log = self.model.predict(input_data)[0]
            
            # Reverse log-transform
            prediction = np.expm1(pred_log)
            
            return {
                'prediction': float(prediction),
                'prediction_log': float(pred_log),
                'success': True,
                'input': {
                    'industry': industry,
                    'city': city,
                    'investment_type': investment_type,
                    'subvertical': subvertical
                }
            }
            
        except Exception as e:
            return {
                'prediction': 0.0,
                'error': str(e),
                'success': False
            }
    
    def predict_batch(self, inputs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Predict funding for multiple startups
        
        Args:
            inputs: List of dictionaries with industry, city, investment_type, (optional) subvertical
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for input_data in inputs:
            result = self.predict(
                industry=input_data.get('industry', 'Unknown'),
                city=input_data.get('city', 'Unknown'),
                investment_type=input_data.get('investment_type', 'Seed Funding'),
                subvertical=input_data.get('subvertical')
            )
            results.append(result)
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.loaded


# Import pandas for DataFrame
import pandas as pd

# Make it importable
__all__ = ['StartupFundingModel']

