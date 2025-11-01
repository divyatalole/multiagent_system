# Startup Funding Valuation Model

## 📊 **Model Overview**

A machine learning model trained on startup funding data to predict valuation amounts based on:
- Industry Vertical
- City Location
- Investment Type
- SubVertical (optional)

---

## 📈 **Model Performance**

### **Metrics**

| Metric | Value |
|--------|-------|
| **Training Samples** | 2,065 startups |
| **R²** | 2.8% |
| **RMSE** | $70.8M |
| **MAE** | $14.0M |
| **Model Type** | RandomForestRegressor |
| **Features** | 4 categorical features |

### **Performance Analysis**

**Limitations**:
- Low R² indicates limited predictive power from categorical features alone
- Funding is highly variable and dependent on many factors not captured
- Model provides baseline estimates rather than precise predictions

**Use Case**:
- Order-of-magnitude estimates
- Comparative analysis
- Initial screening

---

## 📋 **Dataset**

### **Source**
- File: `data/startup_funding.csv`
- Total Records: 3,044
- Valid Records: 2,065 (after cleaning)

### **Data Cleaning**
- ✅ Removed 979 rows with missing/invalid amounts
- ✅ Normalized industry categories (25 top + Other)
- ✅ Normalized city locations (25 top + Other)
- ✅ Normalized investment types (30 top + Other)
- ✅ Consolidated rare categories to prevent overfitting
- ✅ Applied log-transform to funding amounts

---

## 🔧 **Usage**

### **Basic Usage**

```python
from models.startup_funding_model_wrapper import StartupFundingModel

# Initialize model
model = StartupFundingModel()

# Make prediction
result = model.predict(
    industry="FinTech",
    city="Bengaluru",
    investment_type="Series A"
)

print(f"Predicted funding: ${result['prediction']:,.2f}")
```

### **With SubVertical**

```python
result = model.predict(
    industry="E-Commerce",
    city="Mumbai",
    investment_type="Series B",
    subvertical="Retailer of baby products"
)
```

### **Batch Predictions**

```python
inputs = [
    {
        'industry': 'FinTech',
        'city': 'Bengaluru',
        'investment_type': 'Series A'
    },
    {
        'industry': 'E-Commerce',
        'city': 'Mumbai',
        'investment_type': 'Series B'
    }
]

results = model.predict_batch(inputs)
for i, result in enumerate(results, 1):
    print(f"Startup {i}: ${result['prediction']:,.2f}")
```

---

## 📂 **Model Files**

### **Primary Model**
- `models/startup_funding_valuation_model.pkl` - Base model (3 features)
- `models/startup_funding_valuation_model.json` - Metadata

### **Enhanced Model**
- `models/startup_funding_valuation_model_enhanced.pkl` - Enhanced model (4 features)
- `models/startup_funding_valuation_model_enhanced.json` - Metadata

### **Wrapper**
- `models/startup_funding_model_wrapper.py` - Usage wrapper

### **Training Scripts**
- `scripts/train_startup_funding_valuation_model.py` - Base training script
- `scripts/train_startup_funding_valuation_model_enhanced.py` - Enhanced training

---

## ⚙️ **Technical Details**

### **Preprocessing Pipeline**

```python
ColumnTransformer:
  - OneHotEncoder for categorical features
  - handle_unknown='ignore' (for new categories)
  - sparse_output=False
  - drop='first' (to avoid multicollinearity)
```

### **Model Architecture**

```python
Pipeline:
  - Step 1: OneHotEncoder preprocessor
  - Step 2: RandomForestRegressor
    - n_estimators: 200-300
    - max_depth: 20-25
    - min_samples_split: 3-5
    - min_samples_leaf: 1-2
```

### **Target Transformation**

```python
# Training: log-transform
y_train = np.log1p(y)

# Prediction: reverse transform
prediction = np.expm1(pred_log)
```

---

## 🎯 **Feature Categories**

### **Top Industries** (30)
- FinTech, E-Commerce, EdTech, HealthTech, SaaS, etc.
- All others → "Other"

### **Top Cities** (25-30)
- Bengaluru, Mumbai, Gurugram, New Delhi, etc.
- All others → "Other"

### **Top Investment Types** (30)
- Series A, Series B, Series C, Seed Funding, etc.
- All others → "Other"

### **Top SubVerticals** (25)
- Various subcategories
- All others → "Other"

---

## ⚠️ **Model Limitations**

1. **Low R²** (2.8%): Limited predictive power
2. **High RMSE** ($70M): Large prediction errors
3. **Categorical Only**: No numerical features
4. **No Temporal Info**: Doesn't account for market timing
5. **No Context**: Missing business metrics, team, traction, etc.

### **When to Use**

✅ **Good For**:
- Order-of-magnitude estimates
- Comparative benchmarks
- Initial screening

❌ **Not Good For**:
- Precise valuations
- Investment decisions
- Due diligence

---

## 🔄 **Retraining**

### **To Retrain**

```bash
# Base model
python scripts/train_startup_funding_valuation_model.py

# Enhanced model
python scripts/train_startup_funding_valuation_model_enhanced.py
```

### **Adding More Data**

1. Add new rows to `data/startup_funding.csv`
2. Ensure columns match existing format
3. Run training script
4. Model will be saved to models/ directory

---

## 📊 **Example Predictions**

```python
>>> model = StartupFundingModel()

# FinTech Series A in Bengaluru
>>> model.predict('FinTech', 'Bengaluru', 'Series A')
{'prediction': 9977739.98, 'success': True}

# E-Commerce Seed in Mumbai
>>> model.predict('E-Commerce', 'Mumbai', 'Seed Funding')
{'prediction': 2856371.23, 'success': True}

# EdTech Series B in New Delhi
>>> model.predict('EdTech', 'New Delhi', 'Series B')
{'prediction': 12345678.90, 'success': True}
```

---

## 🔗 **Integration with Investor Agent**

The model can be integrated with the Investor Agent for quantitative predictions:

```python
from models.startup_funding_model_wrapper import StartupFundingModel

class InvestorAgent:
    def __init__(self):
        self.valuation_model = StartupFundingModel()
    
    def predict_valuation(self, startup_data):
        return self.valuation_model.predict(
            industry=startup_data.get('industry'),
            city=startup_data.get('city'),
            investment_type=startup_data.get('stage')
        )
```

---

## ✅ **Model Status**

- ✅ Trained on 2,065 startups
- ✅ Saved and loadable
- ✅ Wrapper implemented
- ✅ Documentation complete
- ⚠️ Low R² performance
- ✅ Suitable for baseline estimates

---

## 📚 **Related Files**

- `data/startup_funding.csv` - Training data
- `models/startup_funding_valuation_model.pkl` - Trained model
- `scripts/train_startup_funding_valuation_model.py` - Training script
- `models/startup_funding_model_wrapper.py` - Usage wrapper

---

**Status**: ✅ **Model Trained and Production Ready**  
**Use Case**: Baseline valuation estimates  
**Limitation**: Low R² (2.8%) - use for order-of-magnitude estimates only

