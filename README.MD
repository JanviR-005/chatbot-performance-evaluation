# Chatbot Framework Performance Prediction

## Project Overview
This project develops predictive models to evaluate two critical performance measures for a chatbot framework:
1. **TARGET_Capacity**: The framework's duration for holding substantial conversations
2. **TARGET_CaseCount**: The frequency of AI tasks that can run simultaneously

The project implements multiple regression techniques including Linear Regression, Polynomial Regression, Ridge Regression, and Poisson Regression with comprehensive preprocessing and hyperparameter tuning.

## Dataset
- **Training Dataset**: 2,071 records with 26 features
- **Features**: 23 technical and business parameters including:
  - System metrics (failure rates, latency, utilization)
  - AI-specific metrics (learning stability, adoption index)
  - Security metrics (breach incidents, vulnerability scores)
  - User metrics (active user base, query efficiency)
- **Target Variables**: 
  - TARGET_Capacity (continuous, range: 37.3 - 92.7)
  - TARGET_CaseCount (count data, range: 1 - 57)

## Key Features
- Comprehensive exploratory data analysis (EDA)
- Multicollinearity detection and handling
- Multiple preprocessing techniques (standardization, normalization, PCA)
- Baseline and advanced regression models
- Hyperparameter tuning with GridSearchCV
- Model comparison and evaluation

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone this repository
```bash
git clone <your-repo-url>
cd <project-directory>
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Organize your data structure
```
project/
├── Data.csv
├── requirements.txt
└── Chatbot-Evaluation.ipynb
```

## Usage

### Running the Analysis
```bash
jupyter notebook notebook.ipynb
```

### Workflow Steps
1. **Exploratory Data Analysis**: Distribution analysis, correlation matrix, multicollinearity detection
2. **Data Preprocessing**: Train-test split, standardization, normalization, PCA (95% variance)
3. **Baseline Models**: Linear and Poisson regression without PCA
4. **Advanced Models**: Polynomial regression with regularization
5. **Model Selection**: Hyperparameter tuning and performance comparison
6. **Predictions**: Generate predictions on evaluation dataset

## Model Performance

### TARGET_Capacity Models
| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| **Polynomial Ridge (deg=2)** | **13.93** | **3.73** | **2.79** | **0.831** |
| Polynomial Regression | 13.64 | 3.69 | 2.73 | 0.835 |
| Ridge Regression (L2) | 23.12 | 4.81 | 3.67 | 0.720 |
| Linear Regression | 23.15 | 4.81 | 3.68 | 0.720 |
| Linear Regression (PCA) | 24.63 | 4.96 | 3.76 | 0.702 |

### TARGET_CaseCount Models
| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| Poisson Regression (No PCA) | 22.06 | 4.70 | 3.61 | 0.733 |
| **Poisson Regression (PCA 95%)** | **23.78** | **4.88** | **3.72** | **0.713** |

## Key Findings

### Exploratory Data Analysis
- Both target variables show normal distribution with strong correlation (nearly identical patterns)
- Top correlated features: FundingDiversityIndex (0.66), DatasetDiversityScore (0.62), IntegrationEfficiency (0.53)
- Identified 7 pairs of highly correlated features (correlation > 0.8)
- PCA analysis shows ~6 components explain 70% variance, 14 components for 95%

### Data Preprocessing
- Applied standardization (mean=0, std=1) and min-max normalization (0-1 range)
- PCA reduced dimensions from 23 to 14 features while retaining 95% variance
- Patient-aware data splitting (80% train, 20% test) to prevent data leakage

### Model Selection Rationale

**TARGET_Capacity - Polynomial Ridge (degree=2, α=10.0)**
- Selected for better generalization through regularization
- Balances model complexity with performance
- Reduces overfitting risk compared to standard polynomial regression
- R² of 0.831 indicates 83% of variance explained

**TARGET_CaseCount - Poisson Regression with PCA (95% variance)**
- Appropriate for count data (non-negative integers)
- PCA addresses multicollinearity issues
- Better generalization to unseen data
- R² of 0.713 with reduced dimensionality

## Technical Highlights

### Preprocessing Pipeline
- StandardScaler for feature scaling
- MinMaxScaler for normalization
- PCA for dimensionality reduction (14 components)
- Train-test split with random_state=42 for reproducibility

### Hyperparameter Tuning
- GridSearchCV with 5-fold cross-validation
- Tested alpha values: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- Tested polynomial degrees: [1, 2, 3]
- Tested PCA variance thresholds: [0.8, 0.85, 0.9, 0.95]

### Evaluation Metrics
- **MSE**: Mean Squared Error (penalizes large errors)
- **RMSE**: Root Mean Squared Error (same units as target)
- **MAE**: Mean Absolute Error (average prediction error)
- **R²**: Coefficient of Determination (variance explained)

## Requirements
See `requirements.txt` for detailed dependencies. Main libraries:
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 0.24.0
- statsmodels >= 0.13.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Output Files
- `eval_2_with_predictions.csv`: Evaluation dataset with predicted values for both targets

## Future Improvements
- Explore ensemble methods (Random Forest, Gradient Boosting)
- Implement time-series analysis if temporal patterns exist
- Test deep learning approaches for non-linear relationships
- Perform feature engineering to create interaction terms
- Conduct residual analysis for model diagnostics

## Model Limitations
- Linear models may not fully capture complex non-linear relationships
- PCA components lose interpretability of original features
- Count data modeling could benefit from negative binomial regression for overdispersion
- Limited validation on completely unseen data distributions

---
**Note**: Models should be validated on production data before deployment in real-world chatbot frameworks.
