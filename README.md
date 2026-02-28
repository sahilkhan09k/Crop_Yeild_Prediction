# 🌾 Crop Yield Prediction Using Machine Learning

> **A full-stack machine learning application to predict rice crop yield based on agricultural practices, field conditions, and farming methods — helping farmers and agronomists make data-driven decisions.**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Data Preprocessing](#data-preprocessing)
6. [Machine Learning Pipeline](#machine-learning-pipeline)
7. [Frontend (React + Tailwind)](#frontend-react--tailwind)
8. [Backend (Flask/FastAPI)](#backend-flaskfastapi)
9. [Step-by-Step Project Guide](#step-by-step-project-guide)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

This project aims to predict the **Yield (kg/acre)** of rice crops in Bihar, India, using information collected from farmers at the field level. The data includes details on land preparation, irrigation, fertilizer usage, tillage methods, nursery practices, and harvesting techniques.

### Problem Statement
Crop yield prediction is critical for:
- **Food security planning** at district/state levels
- **Helping farmers** optimize their agricultural practices
- **Policy-making** for agricultural subsidies and resource allocation
- **Early market prediction** for pricing strategies

### Task Type
**Supervised Regression** — Predict a continuous numeric target `Yield` (kg per acre).

---

## 📊 Dataset Description

### Files Overview

| File | Size | Description |
|------|------|-------------|
| `Dataset/Train.csv` | ~1.5 MB | Training data with labeled yield values (3,871 records) |
| `Dataset/Test.csv` | ~508 KB | Test data without labels (for prediction/submission) |
| `Dataset/SampleSubmission.csv` | ~23 KB | Sample output format for competition submission |
| `Dataset/VariableDescription.csv` | 2.8 KB | Column definitions and descriptions |

### Dataset Source / Context
- **Geography:** Bihar state, India (Districts: Nalanda, Gaya, Vaishali, Jamui)
- **Crop:** Rice (Kharif season, sown ~Jun–Aug, harvested ~Oct–Dec)
- **Year:** 2022 growing season
- **Size:** ~3,871 training samples, **18 selected features** (from 43 original columns)

---

### 📦 Selected Features (18 of 43)

After domain analysis and data quality checks, **18 raw columns** were selected as the most predictive. These engineer into **20 model-ready features** (date columns split into derived metrics).

> **Why only 18?** The remaining 25 columns either have >40% missing values, carry redundant information already captured by the selected set, or are administrative fields (nursery factors, threshing dates, residue percentages) that show weak correlation with yield in this dataset.

#### 🌍 Location (2 features)
| # | Column | Why Selected |
|---|--------|--------------|
| 1 | `District` | Geography drives soil quality, water access & infrastructure — Nalanda vs Jamui show ~30% mean yield difference |
| 2 | `Block` | Sub-district granularity; blocks within the same district can vary significantly in canal access & soil type |

#### 🌱 Land & Crop Setup (4 features)
| # | Column | Why Selected |
|---|--------|--------------|
| 3 | `Acre` | Direct denominator of yield calculation — farm size captures farm scale effects |
| 4 | `CropCultLand` | Area actually under crop; differs from `Acre` when part of the land is fallow |
| 5 | `CropEstMethod` | One of the strongest predictors — `Broadcasting` vs `Manual_PuddledRandom` vs `LineSowing` significantly affect plant density and yield |
| 6 | `CropTillageDepth` | Deeper tillage = better root development and water retention; low missingness, clean numeric |

#### 🌱 Sowing (2 raw → 2 engineered features)
| # | Column | Why Selected |
|---|--------|--------------|
| 7 | `SeedlingsPerPit` | Planting density directly controls yield potential — too few or too many seedlings harm output |
| 8 | `SeedingSowingTransplanting` | **Engineered → `SowingDOY` + `CropDuration_days`** — timing of sowing relative to monsoon is critical for rice |

#### 💧 Irrigation (2 features)
| # | Column | Why Selected |
|---|--------|--------------|
| 9 | `TransplantingIrrigationSource` | `Rainfed` vs `Boring` vs `Canal` — the most impactful irrigation factor; Rainfed farmers are fully weather-dependent |
| 10 | `StandingWater` | Days of standing water is a key rice-specific variable; rice requires flooded conditions for optimal growth |

#### 🧪 Chemical Fertilizer (4 features)
| # | Column | Why Selected |
|---|--------|--------------|
| 11 | `NoFertilizerAppln` | Number of applications = measure of farmer care intensity; 3 applications consistently outperforms 1 |
| 12 | `BasalDAP` | DAP at land prep provides phosphorus + nitrogen for root establishment — critical foundation |
| 13 | `BasalUrea` | Basal nitrogen application sets the early growth trajectory |
| 14 | `1tdUrea` | First top-dress urea — timed at tillering stage, strongest individual nitrogen dose affecting grain count |

#### 🌾 Harvest (2 raw → 3 engineered features)
| # | Column | Why Selected |
|---|--------|--------------|
| 15 | `Harv_method` | Machine harvest is faster and captures higher yield; hand harvest in very small plots |
| 16 | `Threshing_method` | Machine threshing = lower post-harvest losses; strongly correlated with `Harv_method` but adds independent signal |
| 17 | `Harv_date` | **Engineered → `HarvestDOY`** — harvest month (Oct vs Dec) affects grain drying conditions |
| 18 | `2tdUrea` | Second top-dress urea — applied at panicle initiation stage, directly boosts grain filling |

#### 🎯 Target Variable
| Column | Description |
|--------|-------------|
| `Yield` | **Crop yield in kg per acre** (target to predict) |

---

### 🔢 Columns Dropped & Why

| Dropped Column(s) | Reason |
|-------------------|--------|
| `ID` | Unique identifier — zero predictive value |
| `CultLand` | Redundant — `CropCultLand` and `Acre` carry the same crop-specific area signal |
| `LandPreparationMethod` | Multi-label with 4+ values; signal mostly captured by `CropTillageDepth` + `CropEstMethod` |
| `CropTillageDate` | Tillage date signal largely overlaps with `SeedingSowingTransplanting` |
| `RcNursEstDate` | >35% missing; nursery date signal is already carried by `SeedingSowingTransplanting` |
| `NursDetFactor`, `TransDetFactor` | Multi-label reasoning fields; subjective farmer-reported factors with no direct yield link |
| `TransplantingIrrigationHours`, `TransIrriCost`, `TransplantingIrrigationPowerSource` | Collinear with `TransplantingIrrigationSource`; cost/hours are noisy proxies |
| `OrgFertilizers`, `Ganaura`, `CropOrgFYM`, `PCropSolidOrgFertAppMethod` | >40% missing; organic fertilizer is used by <30% of farmers — sparse signal |
| `CropbasalFerts`, `MineralFertAppMethod`, `FirstTopDressFert` | Type/method columns — the *amount* (DAP, Urea kg) carries more quantitative signal |
| `1appDaysUrea`, `2appDaysUrea` | Timing of dose intervals; noisy and high missingness; less impactful than dose amounts |
| `Harv_hand_rent`, `Threshing_date` | Cost/date proxies; collinear with `Harv_method` and `Threshing_method` |
| `Residue_length`, `Residue_perc`, `Stubble_use` | Post-harvest residue management has negligible effect on current season's yield |

---

### 📈 Dataset Statistics
- **Total Training Samples:** ~3,871
- **Selected Raw Features:** 18 columns
- **Final Model Features:** ~20 (after date engineering)
- **Numeric Features (selected):** 8 (`Acre`, `CropCultLand`, `CropTillageDepth`, `SeedlingsPerPit`, `StandingWater`, `BasalDAP`, `BasalUrea`, `1tdUrea`, `2tdUrea`)
- **Categorical Features (selected):** 6 (`District`, `Block`, `CropEstMethod`, `TransplantingIrrigationSource`, `Harv_method`, `Threshing_method`)
- **Date Features (selected):** 2 → engineered into 3 (`SowingDOY`, `CropDuration_days`, `HarvestDOY`)
- **Target Range:** ~4 to 3,600+ kg/acre (right-skewed → log transformed)
- **Missing Values:** Present mainly in fertilizer columns — handled via median/zero-fill

---

## 🛠️ Tech Stack

### 🐍 Backend / ML
| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core programming language |
| **Pandas** | Data manipulation and cleaning |
| **NumPy** | Numerical computations |
| **Scikit-learn** | ML model building, preprocessing, evaluation |
| **XGBoost** | Gradient boosted trees (primary model) |
| **LightGBM** | Fast gradient boosting (secondary model) |
| **CatBoost** | Categorical feature-aware boosting |
| **Matplotlib / Seaborn** | Data visualization & EDA |
| **Plotly** | Interactive visualizations |
| **Scipy** | Statistical analysis |
| **Joblib / Pickle** | Model serialization |
| **Flask / FastAPI** | REST API server to serve predictions |
| **Jupyter Notebook** | Exploration and prototyping |

### ⚛️ Frontend
| Technology | Purpose |
|-----------|---------|
| **React 18** | Core frontend framework |
| **Tailwind CSS** | Utility-first styling |
| **Axios** | HTTP client for API calls |
| **React Charts (Recharts)** | Data visualizations |
| **React Hook Form** | Form state management |
| **React Router DOM** | Client-side routing |

### ☁️ DevOps & Tooling
| Technology | Purpose |
|-----------|---------|
| **Git + GitHub** | Version control |
| **Jupyter Notebooks** | EDA and model building |
| **VS Code** | IDE |
| **Postman** | API testing |
| **Vite** | Frontend build tool |

---

## 📁 Project Structure

```
Crop_Yeild_Prediction/
│
├── Dataset/
│   ├── Train.csv
│   ├── Test.csv
│   ├── SampleSubmission.csv
│   └── VariableDescription.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb        # Data cleaning and feature engineering
│   ├── 03_Model_Training.ipynb       # Model training and comparison
│   └── 04_Model_Evaluation.ipynb     # Evaluation metrics and visualizations
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py                # Missing value handling, outlier removal
│   │   ├── encoder.py                # Categorical encoding
│   │   └── feature_engineer.py      # Feature creation and transformation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                  # Model training scripts
│   │   ├── predict.py                # Inference script
│   │   └── evaluate.py              # Evaluation utilities
│   │
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py
│       └── config.py                # Configuration and hyperparameters
│
├── api/
│   ├── app.py                        # Flask/FastAPI app
│   ├── routes.py                     # API endpoints
│   └── schemas.py                    # Request/Response schemas
│
├── models/
│   ├── xgboost_model.pkl             # Saved XGBoost model
│   ├── lgbm_model.pkl                # Saved LightGBM model
│   └── preprocessor.pkl             # Saved preprocessing pipeline
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── PredictionForm.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   ├── FeatureImportance.jsx
│   │   │   └── Navbar.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Predict.jsx
│   │   │   └── About.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── package.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 Data Preprocessing

### Step 1: Load the Data
```python
import pandas as pd
train = pd.read_csv("Dataset/Train.csv")
test  = pd.read_csv("Dataset/Test.csv")
```

### Step 2: Initial Inspection
```python
train.info()
train.describe()
train.isnull().sum().sort_values(ascending=False)
```

### Step 3: Select & Drop Columns
```python
# Keep only the 18 selected features + target
SELECTED_FEATURES = [
    # Location
    'District', 'Block',
    # Land & Crop Setup
    'Acre', 'CropCultLand', 'CropEstMethod', 'CropTillageDepth',
    # Sowing
    'SeedlingsPerPit', 'SeedingSowingTransplanting',
    # Irrigation
    'TransplantingIrrigationSource', 'StandingWater',
    # Chemical Fertilizer
    'NoFertilizerAppln', 'BasalDAP', 'BasalUrea', '1tdUrea', '2tdUrea',
    # Harvest
    'Harv_method', 'Threshing_method', 'Harv_date',
    # Target
    'Yield'
]
train = train[SELECTED_FEATURES]
test  = test[[c for c in SELECTED_FEATURES if c != 'Yield']]
```

### Step 4: Handle Missing Values

| Strategy | Columns |
|---------|---------|
| **Median imputation** | `BasalDAP`, `BasalUrea`, `1tdUrea`, `2tdUrea`, `StandingWater`, `SeedlingsPerPit`, `CropTillageDepth` |
| **Zero-fill** | `NoFertilizerAppln` (no applications = 0, not unknown) |
| **Mode imputation** | `CropEstMethod`, `Harv_method`, `Threshing_method`, `TransplantingIrrigationSource` |
| **Date forward-fill** | `SeedingSowingTransplanting`, `Harv_date` |

```python
import pandas as pd
import numpy as np

# Numeric: median imputation
numeric_cols = ['BasalDAP', 'BasalUrea', '1tdUrea', '2tdUrea',
                'StandingWater', 'SeedlingsPerPit', 'CropTillageDepth']
for col in numeric_cols:
    train[col] = train[col].fillna(train[col].median())
    test[col]  = test[col].fillna(train[col].median())   # use train median on test

# Categorical: mode imputation
cat_cols = ['CropEstMethod', 'Harv_method', 'Threshing_method',
            'TransplantingIrrigationSource']
for col in cat_cols:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col]  = test[col].fillna(mode_val)

# NoFertilizerAppln: zero fill
train['NoFertilizerAppln'] = train['NoFertilizerAppln'].fillna(0)
test['NoFertilizerAppln']  = test['NoFertilizerAppln'].fillna(0)
```

### Step 5: Feature Engineering from Date Columns
```python
# Parse date columns
train['SeedingSowingTransplanting'] = pd.to_datetime(
    train['SeedingSowingTransplanting'], errors='coerce')
train['Harv_date'] = pd.to_datetime(train['Harv_date'], errors='coerce')

# Engineer new features
train['SowingDOY']       = train['SeedingSowingTransplanting'].dt.dayofyear
train['HarvestDOY']      = train['Harv_date'].dt.dayofyear
train['CropDuration_days'] = (train['Harv_date'] -
                              train['SeedingSowingTransplanting']).dt.days

# Drop raw date columns (replaced by engineered ones)
train.drop(columns=['SeedingSowingTransplanting', 'Harv_date'], inplace=True)

# Apply same to test set
test['SeedingSowingTransplanting'] = pd.to_datetime(
    test['SeedingSowingTransplanting'], errors='coerce')
test['Harv_date'] = pd.to_datetime(test['Harv_date'], errors='coerce')
test['SowingDOY']         = test['SeedingSowingTransplanting'].dt.dayofyear
test['HarvestDOY']        = test['Harv_date'].dt.dayofyear
test['CropDuration_days'] = (test['Harv_date'] -
                              test['SeedingSowingTransplanting']).dt.days
test.drop(columns=['SeedingSowingTransplanting', 'Harv_date'], inplace=True)
```

> After this step the model sees **20 features**: 17 original + 3 engineered (`SowingDOY`, `HarvestDOY`, `CropDuration_days`).

### Step 6: Encode Categorical Columns
```python
from sklearn.preprocessing import LabelEncoder

cat_encode_cols = [
    'District', 'Block', 'CropEstMethod',
    'TransplantingIrrigationSource', 'Harv_method', 'Threshing_method'
]

le = LabelEncoder()
for col in cat_encode_cols:
    train[col] = le.fit_transform(train[col].astype(str))
    # Encode test using same categories (handle unseen with try/except)
    test[col]  = test[col].map(
        dict(zip(le.classes_, le.transform(le.classes_)))
    ).fillna(-1).astype(int)
```

### Step 7: Outlier Capping on Target
```python
# Winsorize: cap extreme yield values at 1st–99th percentile
lower = train['Yield'].quantile(0.01)
upper = train['Yield'].quantile(0.99)
train['Yield'] = np.clip(train['Yield'], lower, upper)
```

### Step 8: Log-Transform Target Variable
```python
train['Yield_log'] = np.log1p(train['Yield'])  # log(1 + Yield) to handle zeros
# At inference time: predicted_yield = np.expm1(model.predict(X))
```

### Step 9: No Scaling Needed (Tree Models)
XGBoost, LightGBM, CatBoost, and Random Forest are **scale-invariant** — they split on thresholds, not distances. Scaling is skipped.

> Apply `StandardScaler` only if running the Ridge Regression baseline.

### Step 10: Final Feature Set & Train-Validation Split
```python
from sklearn.model_selection import train_test_split

# Final 20 features going into the model
FINAL_FEATURES = [
    'District', 'Block',
    'Acre', 'CropCultLand', 'CropEstMethod', 'CropTillageDepth',
    'SeedlingsPerPit', 'SowingDOY',
    'TransplantingIrrigationSource', 'StandingWater',
    'NoFertilizerAppln', 'BasalDAP', 'BasalUrea', '1tdUrea', '2tdUrea',
    'Harv_method', 'Threshing_method',
    'HarvestDOY', 'CropDuration_days'
]

X = train[FINAL_FEATURES]
y = train['Yield_log']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {X_train.shape[1]} features, {X_train.shape[0]} samples")
```

---

## 🤖 Machine Learning Pipeline

### Recommended Models (in Priority Order)

| Model | Reason |
|-------|--------|
| **XGBoost Regressor** | Best overall performance on tabular data, handles missing values natively |
| **LightGBM Regressor** | Faster training, good for large datasets |
| **CatBoost Regressor** | Built-in categorical feature handling |
| **Random Forest Regressor** | Strong baseline, interpretable |
| **Gradient Boosting Regressor** | Classic ensemble method |
| **Ridge / Lasso Regression** | Simple linear baseline |

---

### Step 1: Baseline Models
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

models = {
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=300, learning_rate=0.05, verbose=0)
}

from sklearn.metrics import mean_squared_error
import numpy as np

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"{name}: RMSE = {rmse:.4f}")
```

---

### Step 2: Hyperparameter Tuning
Use **Optuna** or **GridSearchCV** for tuning:

```python
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    model = XGBRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5,
                              scoring='neg_root_mean_squared_error')
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

---

### Step 3: Cross-Validation
```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    best_model, X, y,
    cv=kf,
    scoring='neg_root_mean_squared_error'
)
print(f"CV RMSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

### Step 4: Feature Importance
```python
import matplotlib.pyplot as plt

feature_importances = pd.Series(
    best_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

feature_importances.head(20).plot(kind='barh', figsize=(10, 8))
plt.title('Feature Importances (All 20 Features)')
plt.tight_layout()
plt.show()
```

---

### Step 5: Ensemble / Stacking (Optional Advanced)
```python
from sklearn.ensemble import StackingRegressor

estimators = [
    ('xgb', XGBRegressor(**best_params_xgb)),
    ('lgbm', LGBMRegressor(**best_params_lgbm)),
]
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5
)
stacking_model.fit(X_train, y_train)
```

---

### Step 6: Save Model
```python
import joblib

joblib.dump(best_model, 'models/xgboost_model.pkl')
joblib.dump(preprocessor_pipeline, 'models/preprocessor.pkl')
```

---

## 🖥️ Frontend (React + Tailwind)

### Setup
```bash
cd frontend
npm create vite@latest . -- --template react
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install axios recharts react-router-dom react-hook-form
```

### Key Pages & Components

| Component | Description |
|-----------|-------------|
| `Home.jsx` | Landing page with project overview and quick stats |
| `Predict.jsx` | Interactive form to enter farm details and get yield prediction |
| `Dashboard.jsx` | Visualizations: yield distribution, feature importance charts |
| `PredictionForm.jsx` | Multi-step form with field validation |
| `ResultCard.jsx` | Displays prediction result with confidence range |
| `Navbar.jsx` | Navigation bar |

### API Integration
```jsx
// src/components/PredictionForm.jsx
import axios from 'axios';

const handleSubmit = async (formData) => {
  const response = await axios.post('http://localhost:5000/predict', formData);
  setPredictedYield(response.data.yield);
};
```

---

## 🔌 Backend (Flask/FastAPI)

### Flask App
```python
# api/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load('models/xgboost_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df_processed = preprocessor.transform(df)
    log_yield = model.predict(df_processed)[0]
    yield_pred = float(np.expm1(log_yield))
    return jsonify({'yield': round(yield_pred, 2), 'unit': 'kg/acre'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## 📋 Step-by-Step Project Guide

### Phase 1: Setup & Environment
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Initialize Git repository
- [ ] Set up folder structure as described above

### Phase 2: Exploratory Data Analysis (EDA)
- [ ] Load and inspect dataset shape, dtypes, and nulls
- [ ] Visualize `Yield` distribution (histogram, boxplot)
- [ ] Visualize `Yield` by `District`, `Block`, `CropEstMethod`
- [ ] Correlation heatmap for numeric features
- [ ] Analyze date feature distributions (sowing month, harvest month)
- [ ] Identify and visualize outliers in `Yield`
- [ ] Explore multi-label categorical fields (count unique combinations)
- [ ] Pairplot of top numeric features vs Yield

### Phase 3: Data Preprocessing
- [ ] Select the 18 chosen columns + `Yield` target; drop all others
- [ ] Median-impute numeric columns (`BasalDAP`, `BasalUrea`, `1tdUrea`, `2tdUrea`, `StandingWater`, `SeedlingsPerPit`, `CropTillageDepth`)
- [ ] Zero-fill `NoFertilizerAppln`; mode-fill categorical cols
- [ ] Parse `SeedingSowingTransplanting` and `Harv_date` → engineer `SowingDOY`, `HarvestDOY`, `CropDuration_days`
- [ ] Label-encode 6 categorical columns (`District`, `Block`, `CropEstMethod`, `TransplantingIrrigationSource`, `Harv_method`, `Threshing_method`)
- [ ] Cap yield outliers at 1st–99th percentile (winsorization)
- [ ] Apply `np.log1p()` transformation to `Yield`
- [ ] Build scikit-learn `Pipeline` wrapping imputer + encoder + model
- [ ] Verify final feature matrix shape is `(n_samples, 20)`

### Phase 4: Model Training
- [ ] Train baseline models (Ridge, Random Forest)
- [ ] Train XGBoost, LightGBM, CatBoost
- [ ] Evaluate all models on validation set (RMSE, MAE, R²)
- [ ] Tune best model using Optuna or GridSearchCV
- [ ] Perform 5-fold cross-validation on best model
- [ ] Analyze feature importances
- [ ] (Optional) Build stacking ensemble
- [ ] Save final model and preprocessor with joblib

### Phase 5: Backend API
- [ ] Set up Flask or FastAPI project
- [ ] Load trained model and preprocessor
- [ ] Implement `/predict` POST endpoint
- [ ] Implement `/health` GET endpoint
- [ ] Test API with Postman
- [ ] Enable CORS for React frontend

### Phase 6: Frontend Development
- [ ] Bootstrap React + Vite + Tailwind project
- [ ] Build `Navbar` component
- [ ] Build `Home` landing page
- [ ] Build `PredictionForm` with all input fields
- [ ] Build `ResultCard` to show prediction output
- [ ] Integrate Axios calls to Flask API
- [ ] Build `Dashboard` with Recharts visualizations
- [ ] Make app fully responsive

### Phase 7: Testing & Validation
- [ ] Unit test preprocessing functions
- [ ] Integration test the API endpoints
- [ ] End-to-end test from form submission to result display
- [ ] Validate prediction outputs on test set and compare with `SampleSubmission.csv`

### Phase 8: Deployment (Optional)
- [ ] Containerize API with Docker
- [ ] Deploy Flask backend to Render / Railway / AWS
- [ ] Deploy React frontend to Vercel / Netlify
- [ ] Connect frontend to deployed API URL

---

## 📏 Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **RMSE** | √(Σ(y - ŷ)²/n) | Penalizes large errors; primary metric |
| **MAE** | Σ\|y - ŷ\|/n | Robust to outliers; average error magnitude |
| **R² Score** | 1 - SS_res/SS_tot | Explained variance; closer to 1 is better |
| **MAPE** | Σ\|(y - ŷ)/y\|/n × 100 | Percentage error; useful for communication |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate(y_true, y_pred_log):
    y_pred = np.expm1(y_pred_log)       # reverse log transform
    y_true_orig = np.expm1(y_true)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred))
    mae  = mean_absolute_error(y_true_orig, y_pred)
    r2   = r2_score(y_true_orig, y_pred)
    print(f"RMSE: {rmse:.2f} kg/acre")
    print(f"MAE : {mae:.2f} kg/acre")
    print(f"R²  : {r2:.4f}")
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Crop_Yield_Prediction.git
cd Crop_Yield_Prediction
```

### 2. Install Python Dependencies
```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Run Jupyter Notebooks (EDA & Model Training)
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 4. Start Backend API
```bash
cd api
python app.py
# API runs at http://localhost:5000
```

### 5. Start Frontend
```bash
cd frontend
npm install
npm run dev
# App runs at http://localhost:5173
```

---

## 📦 requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
optuna>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
flask>=3.0.0
flask-cors>=4.0.0
joblib>=1.3.0
jupyter>=1.0.0
scipy>=1.11.0
```

---

## 🔮 Future Improvements

- [ ] **Add weather data** (rainfall, temperature, humidity) as additional features for richer predictions
- [ ] **Time series integration** — predict yield trends across future seasons
- [ ] **Geo-spatial analysis** — map yield predictions by district using Folium or Leaflet.js
- [ ] **SHAP explanations** — provide farmers with interpretable, per-field explanations of yield factors
- [ ] **Crop variety feature** — include rice variety as a strong predictor
- [ ] **Mobile-responsive PWA** — convert the React app to a Progressive Web App for rural smartphone users
- [ ] **SMS/WhatsApp notifications** — send yield predictions directly to farmers via Twilio API
- [ ] **Batch prediction** — allow uploading CSV of multiple fields for bulk prediction

---

## 🧑‍💻 Author

> **Project:** Crop Yield Prediction Using Machine Learning  
> **Role:** Full-stack ML Developer  
> **Stack:** Python · Scikit-learn · XGBoost · React · Tailwind CSS · Flask

---

## 📄 License

This project is for academic and research purposes. Dataset sourced from agricultural survey data in Bihar, India.
