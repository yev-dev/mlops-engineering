# End-to-End ML Pipeline Tutorial

> **A comprehensive walkthrough of the model development lifecycle using the MLOps Engineering project architecture**

This tutorial covers every stage of the machine learning pipeline — from raw data ingestion to production monitoring — using the practical example of stock price prediction (Exxon Mobil stock based on oil prices) that powers the [mlops-engineering](https://github.com/yev-dev/mlops-engineering) project.

The tutorial is designed to be executed as a Jupyter Notebook. Each section includes runnable Python code so you can follow along interactively.

---

## Table of Contents

1. [The ML Lifecycle Overview](#1-the-ml-lifecycle-overview)
2. [Data Engineering](#2-data-engineering)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Selection](#4-model-selection)
5. [Model Evaluation Methods](#5-model-evaluation-methods)
6. [Model Optimisation Techniques](#6-model-optimisation-techniques)
7. [Data & Concept Drift (Evidently + SHAP)](#7-data--concept-drift-evidently--shap)

---

## 1. The ML Lifecycle Overview

### 1.1 What is an End-to-End ML Pipeline?

A production ML pipeline is not just about training a model — it encompasses the entire journey from business problem definition to ongoing monitoring in production.

```
Business Problem → Data Engineering → Feature Engineering → Model Selection
    → Training & Evaluation → Optimisation → Deployment → Monitoring → Retrain
```

The **mlops-engineering** project implements exactly this lifecycle for a regression use case: predicting Exxon Mobil's stock price from crude oil prices. The architecture is:

| Stage | Implementation |
|---|---|
| **Model Training** | Linear Regression (scikit-learn), serialised with `pickle` |
| **API Service** | FastAPI serving predictions via REST `/predict` endpoint |
| **Containerisation** | Docker + docker-compose |
| **Monitoring** | Prometheus metrics (latency, prediction distribution, error rates) |
| **Dashboards** | Grafana dashboards for visual monitoring |
| **Data Drift** | Evidently (work in progress) |
| **Feature Store / Registry** | MLflow (planned) |
| **Object Storage** | MinIO (planned) |

### 1.2 The Use Case

We train a **linear regression model** to predict the closing price of Exxon Mobil (XOM) using West Texas Intermediate (WTI) crude oil prices as the single feature. The pre-trained model is saved as a `pickle` file and loaded into a FastAPI service for inference.

```python
# Example: Loading the trained model
import pickle
import pandas as pd
import numpy as np

# Load the model (same approach used in app/src/main.py)
with open("regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Make a prediction
oil_price = [[85.0]]  # WTI crude oil price
predicted_stock = model.predict(oil_price)
print(f"Predicted XOM price at oil=${oil_price[0][0]}: ${predicted_stock[0]:.2f}")
```

---

## 2. Data Engineering

Data engineering is the foundation of any ML pipeline. It encompasses the collection, validation, cleaning, and transformation of raw data into a format suitable for modelling.

### 2.1 The Data Pipeline

In the mlops-engineering project, two primary data sources are used:

| Source | Description |
|---|---|
| **Oil Prices (WTI)** | Historical crude oil spot prices (independent variable / feature) |
| **Stock Prices (XOM)** | Historical Exxon Mobil adjusted close prices (dependent variable / target) |

### 2.2 Data Collection

Data can be fetched from public APIs such as `yfinance` (used in the project) or `pandas-datareader`.

```python
import yfinance as yf
import pandas as pd

# Define tickers and date range
oil_ticker = "CL=F"       # Crude Oil Futures
stock_ticker = "XOM"      # Exxon Mobil
start_date = "2015-01-01"
end_date = "2025-01-01"

# Fetch data
oil_data = yf.download(oil_ticker, start=start_date, end=end_date)
xom_data = yf.download(stock_ticker, start=start_date, end=end_date)

# Extract adjusted close prices
oil_prices = oil_data["Adj Close"].rename("WTI_Oil")
xom_prices = xom_data["Adj Close"].rename("XOM_Stock")

# Combine into a single DataFrame
df = pd.concat([oil_prices, xom_prices], axis=1).dropna()
print(df.head())
print(f"Dataset shape: {df.shape}")
```

### 2.3 Data Validation & Quality Checks

Before any modelling, data quality must be verified:

```python
def validate_data(df):
    checks = {}
    
    # Check for missing values
    checks["missing_values"] = df.isnull().sum()
    
    # Check for duplicates
    checks["duplicate_rows"] = df.duplicated().sum()
    
    # Basic statistics
    checks["summary_stats"] = df.describe()
    
    # Check for extreme outliers (beyond 3 sigma)
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
        checks[f"outliers_3sigma_{col}"] = len(outliers)
    
    return checks

quality_report = validate_data(df)
for check, result in quality_report.items():
    print(f"{check}:\n{result}\n")
```

### 2.4 Train/Test Splitting

Data must be split **temporally** (not randomly) for time-series data to avoid look-ahead bias:

```python
from sklearn.model_selection import train_test_split

# For time series, use chronological split
X = df[["WTI_Oil"]].values
y = df["XOM_Stock"].values

# Use 80% for training, 20% for testing (chronological)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)} ({df.index[split_idx].date()})")
print(f"Testing samples:  {len(X_test)} ({df.index[-1].date()})")
```

**Key data engineering principles applied in this project:**

- **Reproducibility**: Data versioning and consistent splits ensure experiments are repeatable
- **Validation**: Quality gates catch data issues early
- **Temporal awareness**: Time-series data is split chronologically, not randomly
- **Serialisation**: Pre-processed datasets can be saved as Parquet or CSV for reuse

---

## 3. Feature Engineering

Feature engineering transforms raw data into predictive signals that improve model performance. Even with a simple linear model, thoughtful feature engineering is critical.

### 3.1 Raw Feature

The simplest feature is the raw oil price — this is what the current mlops-engineering model uses.

```python
# Raw feature: WTI oil price
# The model learns: XOM_price = β₀ + β₁ × WTI_price
```

### 3.2 Lagged Features

Financial time series often depend on past values. We can create lagged features:

```python
def create_lagged_features(df, target_col, lags=[1, 2, 3, 5, 10, 21]):
    """Create lagged features for the target variable."""
    df_feat = df.copy()
    for lag in lags:
        df_feat[f"{target_col}_lag_{lag}"] = df_feat[target_col].shift(lag)
    return df_feat

# Create lagged features
df_lagged = create_lagged_features(df, "WTI_Oil", lags=[1, 2, 5, 21])

# Drop NaN rows from shifting
df_lagged = df_lagged.dropna()
print(f"Lagged features shape: {df_lagged.shape}")
print(df_lagged.head())
```

### 3.3 Rolling Window Features (Moving Averages)

Rolling statistics capture trends and volatility:

```python
def create_rolling_features(df, col, windows=[5, 10, 21, 63]):
    """Create rolling window features."""
    df_feat = df.copy()
    for w in windows:
        df_feat[f"{col}_rolling_mean_{w}d"] = df_feat[col].rolling(w).mean()
        df_feat[f"{col}_rolling_std_{w}d"] = df_feat[col].rolling(w).std()
        df_feat[f"{col}_rolling_min_{w}d"] = df_feat[col].rolling(w).min()
        df_feat[f"{col}_rolling_max_{w}d"] = df_feat[col].rolling(w).max()
    return df_feat

df_features = create_rolling_features(df_lagged, "WTI_Oil")
df_features = df_features.dropna()
print(f"Feature-engineered shape: {df_features.shape}")
```

### 3.4 Price Returns and Ratios

Financial features are often more informative as returns rather than raw prices:

```python
def create_financial_features(df, price_col):
    """Create financial features: returns, log returns, spread."""
    df_feat = df.copy()
    
    # Simple returns
    df_feat[f"{price_col}_return_1d"] = df_feat[price_col].pct_change(1)
    df_feat[f"{price_col}_return_5d"] = df_feat[price_col].pct_change(5)
    df_feat[f"{price_col}_return_21d"] = df_feat[price_col].pct_change(21)
    
    # Log returns (better statistical properties)
    df_feat[f"{price_col}_log_return_1d"] = np.log(df_feat[price_col] / df_feat[price_col].shift(1))
    
    # Price ratio
    df_feat["Oil_Stock_Ratio"] = df_feat["WTI_Oil"] / df_feat["XOM_Stock"]
    
    return df_feat

df_fin = create_financial_features(df_features.drop(columns=["XOM_Stock"]), "WTI_Oil")
df_fin["XOM_Stock"] = df_features["XOM_Stock"]  # re-attach target
df_fin = df_fin.dropna()
print(f"Financial features shape: {df_fin.shape}")
```

### 3.5 Feature Selection

Not all features are useful. We can use correlation analysis and statistical tests:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation with target
correlations = df_fin.corr()["XOM_Stock"].sort_values(ascending=False)
print("Top features correlated with XOM price:")
print(correlations.head(10))

# Visualise correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df_fin.corr(), cmap="RdBu_r", center=0, vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
```

**Feature engineering principles used:**

- **Domain knowledge**: Financial features (returns, moving averages) capture known market behaviours
- **Temporal features**: Lags and rolling windows respect the sequential nature of time series
- **Feature selection**: High-correlation features are retained; redundant features are dropped
- **Stationarity**: Log returns are preferred over raw prices because they are more stationary

---

## 4. Model Selection

Model selection involves choosing the right algorithm based on the problem type, data characteristics, and operational constraints.

### 4.1 Problem Type: Regression

Our use case is a **regression** problem (predicting a continuous stock price). The mlops-engineering project uses a simple **Linear Regression** model, but in practice several models should be compared.

### 4.2 Candidate Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    "SVR": SVR(kernel="rbf"),
}

results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "MSE": mse,
        "MAE": mae,
        "R²": r2,
        "Train Time (s)": f"{train_time:.3f}"
    })

results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
print(results_df)
```

### 4.3 Model Complexity vs. Interpretability Trade-off

| Model | Interpretability | Performance | Training Speed | When to Use |
|---|---|---|---|---|
| **Linear Regression** | ★★★★★ | ★★☆☆☆ | ★★★★★ | Baseline, highly interpretable scenarios |
| **Ridge / Lasso** | ★★★★☆ | ★★★☆☆ | ★★★★★ | When multicollinearity exists |
| **Random Forest** | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | Non-linear relationships, large datasets |
| **Gradient Boosting** | ★☆☆☆☆ | ★★★★★ | ★★☆☆☆ | When maximum accuracy is needed |
| **SVR** | ★☆☆☆☆ | ★★★★☆ | ★☆☆☆☆ | Small to medium datasets, non-linear patterns |

The mlops-engineering project chose **Linear Regression** for its simplicity, interpretability, and low latency in production — critical for a real-time API service.

### 4.4 Cross-Validation for Time Series

Standard k-fold cross-validation leaks future information into training folds. Use **TimeSeriesSplit**:

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

tscv = TimeSeriesSplit(n_splits=5)

model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")

print(f"Time-series CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 4.5 Model Serialisation

The selected model is saved using `pickle` (as done in the mlops-engineering project):

```python
import pickle

# Train final model
final_model = LinearRegression()
final_model.fit(X_train, y_train)

# Serialise
with open("regression_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

print("Model saved to regression_model.pkl")

# Verify load
with open("regression_model.pkl", "rb") as f:
    loaded = pickle.load(f)
    
# Test inference matches
assert np.isclose(final_model.predict([[80]])[0], loaded.predict([[80]])[0])
print("Model serialisation verified ✓")
```

---

## 5. Model Evaluation Methods

Proper evaluation goes beyond a single metric. We need a comprehensive understanding of model behaviour. The evaluation methodology differs fundamentally between **linear models** (parametric, interpretable, strong assumptions) and **cluster-based models** (unsupervised, pattern-discovery, no ground truth).

### 5.1 Regression Metrics (Shared)

These apply to any supervised regression model, including both linear and non-linear models:

```python
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)

y_pred = final_model.predict(X_test)

metrics = {
    "MSE": mean_squared_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "MAPE (%)": mean_absolute_percentage_error(y_test, y_pred) * 100,
    "R²": r2_score(y_test, y_pred),
    "Explained Variance": explained_variance_score(y_test, y_pred),
}

print("Model Evaluation Metrics:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
```

#### Understanding the Metrics

| Metric | Range | Interpretation |
|---|---|---|
| **MSE** | [0, ∞) | Penalises large errors quadratically; sensitive to outliers |
| **RMSE** | [0, ∞) | Same units as target; most commonly reported |
| **MAE** | [0, ∞) | Average absolute error; robust to outliers |
| **MAPE** | [0, ∞) | Percentage error; scale-independent but undefined at y=0 |
| **R²** | (-∞, 1] | Proportion of variance explained. 1 = perfect, 0 = mean-baseline, < 0 = worse than mean |
| **Explained Variance** | (-∞, 1] | Similar to R² but does not assume zero-mean residuals |

### 5.2 Residual Analysis

Residuals (errors = actual − predicted) should be normally distributed with zero mean and constant variance. This is especially important for linear models where the Gauss-Markov theorem assumes homoscedasticity and zero-mean errors.

```python
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Residual distribution
axes[0].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
axes[0].axvline(x=0, color="red", linestyle="--", label="Zero error")
axes[0].set_title("Residual Distribution")
axes[0].set_xlabel("Prediction Error ($)")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Residuals vs. Predicted
axes[1].scatter(y_pred, residuals, alpha=0.5)
axes[1].axhline(y=0, color="red", linestyle="--")
axes[1].set_title("Residuals vs. Predicted")
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residual ($)")

# Q-Q plot for normality
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot (Normality Check)")

plt.tight_layout()
plt.show()

# Statistical test for normality
_, p_value = stats.normaltest(residuals)
print(f"D'Agostino-Pearson normality test p-value: {p_value:.4f}")
print("Residuals are normally distributed" if p_value > 0.05 else "Residuals are NOT normally distributed")
```

### 5.3 Temporal Performance

For time-series models, we must check if performance decays over time:

```python
def temporal_performance_analysis(y_actual, y_predicted, timestamps, window=63):
    """Analyse model performance over time using rolling windows."""
    df_perf = pd.DataFrame({
        "timestamp": timestamps,
        "actual": y_actual,
        "predicted": y_predicted
    })
    
    df_perf["error"] = df_perf["actual"] - df_perf["predicted"]
    df_perf["abs_error"] = df_perf["error"].abs()
    
    # Rolling MAE over time
    df_perf["rolling_mae"] = df_perf["abs_error"].rolling(window).mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_perf["timestamp"], df_perf["actual"], label="Actual", alpha=0.7)
    ax.plot(df_perf["timestamp"], df_perf["predicted"], label="Predicted", alpha=0.7)
    ax.fill_between(df_perf["timestamp"], 
                     df_perf["predicted"] - df_perf["rolling_mae"],
                     df_perf["predicted"] + df_perf["rolling_mae"],
                     alpha=0.2, label=f"±{window}-day Rolling MAE")
    ax.set_title("Temporal Performance: Actual vs. Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price ($)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return df_perf

# Create timestamps for test set
test_timestamps = df.index[split_idx:]
perf_df = temporal_performance_analysis(y_test, y_pred, test_timestamps)
```

### 5.4 Walk-Forward Validation

The gold standard for time-series evaluation is walk-forward (expanding window) validation:

```python
def walk_forward_validation(X, y, model_class, initial_window=500, step=50):
    """
    Walk-forward validation: train on expanding window, test on next period.
    Simulates how the model would perform if retrained periodically.
    """
    n = len(X)
    predictions = np.full(n, np.nan)
    
    for end in range(initial_window, n, step):
        start = 0  # expanding window: always start from beginning
        test_end = min(end + step, n)
        
        # Train on [start:end], predict on [end:test_end]
        model = model_class()
        model.fit(X[start:end], y[start:end])
        predictions[end:test_end] = model.predict(X[end:test_end])
    
    # Remove leading NaN predictions
    valid_idx = ~np.isnan(predictions)
    return predictions[valid_idx], y[valid_idx]

wf_predictions, wf_actual = walk_forward_validation(X, y, LinearRegression)
wf_rmse = np.sqrt(mean_squared_error(wf_actual, wf_predictions))
wf_r2 = r2_score(wf_actual, wf_predictions)

print(f"Walk-Forward RMSE: {wf_rmse:.4f}")
print(f"Walk-Forward R²:   {wf_r2:.4f}")
```

### 5.5 In-Depth Evaluation for Linear Models

Linear models carry strong statistical assumptions. Evaluation must verify these assumptions hold, otherwise predictions and confidence intervals are unreliable.

#### 5.5.1 Coefficient Analysis and Statistical Significance

Unlike tree-based models, linear regression provides interpretable coefficients with formal statistical tests:

```python
import statsmodels.api as sm

# Fit with statsmodels for full statistical output
X_train_sm = sm.add_constant(X_train)  # add intercept
sm_model = sm.OLS(y_train, X_train_sm).fit()
print(sm_model.summary())

# Extract key statistics
print(f"\n--- Coefficient Analysis ---")
print(f"Coefficient (β₁) = {sm_model.params[1]:.4f}")
print(f"Std Error       = {sm_model.bse[1]:.4f}")
print(f"t-statistic     = {sm_model.tvalues[1]:.4f}")
print(f"p-value         = {sm_model.pvalues[1]:.6f}")
print(f"95% CI          = ({sm_model.conf_int()[1][0]:.4f}, {sm_model.conf_int()[1][1]:.4f})")
print(f"\nR² = {sm_model.rsquared:.4f}")
print(f"Adj. R² = {sm_model.rsquared_adj:.4f}")
print(f"F-statistic = {sm_model.fvalue:.4f} (p={sm_model.f_pvalue:.6e})")
```

**What to look for:**
- **p-value < 0.05**: The feature is statistically significant at 95% confidence
- **95% CI width**: Narrower = more precise estimate. If CI contains 0, the feature may be irrelevant
- **Adj. R² vs R²**: Large difference indicates overfitting (too many features)
- **F-statistic p-value**: Tests whether *any* feature explains the target (overall model significance)

#### 5.5.2 Multicollinearity: Variance Inflation Factor (VIF)

When features are correlated, coefficient estimates become unstable and uninterpretable:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create multi-feature DataFrame for VIF analysis
X_multi = pd.DataFrame({
    "WTI_Oil": df["WTI_Oil"].values,
    "WTI_lag1": df["WTI_Oil"].shift(1).values,
    "WTI_lag5": df["WTI_Oil"].shift(5).values,
    "WTI_MA21": df["WTI_Oil"].rolling(21).mean().values,
}).dropna()

vif_data = pd.DataFrame()
vif_data["Feature"] = X_multi.columns
vif_data["VIF"] = [variance_inflation_factor(X_multi.values, i) 
                   for i in range(X_multi.shape[1])]
print(vif_data)
```

**VIF Interpretation:**
| VIF | Signal | Action |
|---|---|---|
| 1 | No correlation | ✓ Ideal |
| 1–5 | Moderate correlation | Acceptable |
| 5–10 | High correlation | Consider removing or regularising (Ridge) |
| >10 | Severe multicollinearity | Remove feature or use Lasso / Ridge |

#### 5.5.3 Heteroscedasticity Tests

Linear models assume constant variance of residuals (homoscedasticity). When violated, standard errors are wrong:

```python
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Breusch-Pagan test
bp_test = het_breuschpagan(residuals, sm.add_constant(X_test))
print(f"Breusch-Pagan Test:")
print(f"  LM statistic: {bp_test[0]:.4f}")
print(f"  p-value:      {bp_test[1]:.4f}")
print(f"  Conclusion: {'Homoscedastic (OK)' if bp_test[1] > 0.05 else 'Heteroscedastic (violation)'}")

# White's test (more general, detects non-linear heteroscedasticity)
white_test = het_white(residuals, sm.add_constant(X_test))
print(f"\nWhite's Test:")
print(f"  LM statistic: {white_test[0]:.4f}")
print(f"  p-value:      {white_test[1]:.4f}")
```

**If heteroscedasticity is detected:**
1. Use **Huber-White robust standard errors** (`HC0`, `HC1`, `HC3` in statsmodels)
2. Transform the target variable (log transformation often stabilises variance)
3. Use weighted least squares

```python
# Fit with robust standard errors
robust_model = sm.OLS(y_train, X_train_sm).fit(cov_type="HC3")
print(robust_model.summary())
```

#### 5.5.4 Autocorrelation in Residuals (Durbin-Watson)

For time-series, residuals should not be autocorrelated:

```python
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_stat:.4f}")
print(f"  DW ≈ 2.0 → No autocorrelation (ideal)")
print(f"  DW < 1.5 → Positive autocorrelation (problematic)")
print(f"  DW > 2.5 → Negative autocorrelation (problematic)")
```

**What to do if autocorrelation exists:**
- Add lagged features of the target (AR terms)
- Use ARIMA or SARIMAX instead of static regression
- Use Newey-West standard errors

```python
from statsmodels.stats.stattools import durbin_watson

# Ljung-Box test for autocorrelation at multiple lags
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print("Ljung-Box Test for autocorrelation:")
print(lb_test)
```

#### 5.5.5 Influence and Leverage Diagnostics

Identify which data points disproportionately drive the model:

```python
from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(sm_model)

# Cook's distance (influence measure)
cooks_d = influence.cooks_distance[0]
print(f"Max Cook's distance: {cooks_d.max():.4f} (threshold ≈ 4/n = {4/len(y_train):.4f})")
print(f"Influential points (> threshold): {(cooks_d > 4/len(y_train)).sum()}")

# Leverage (hat values)
leverage = influence.hat_matrix_diag
print(f"Mean leverage: {leverage.mean():.4f} (expected: p/n = {2/len(y_train):.4f})")
high_leverage = leverage > 2 * leverage.mean()
print(f"High leverage points: {high_leverage.sum()}")

# Plot influence
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].stem(cooks_d, linefmt="C0-", markerfmt="C0o")
axes[0].axhline(y=4/len(y_train), color="red", linestyle="--", label="Threshold")
axes[0].set_title("Cook's Distance")
axes[0].set_xlabel("Observation Index")
axes[0].legend()

axes[1].stem(leverage, linefmt="C1-", markerfmt="C1o")
axes[1].axhline(y=2*leverage.mean(), color="red", linestyle="--", label="2× Mean")
axes[1].set_title("Leverage (Hat Values)")
axes[1].set_xlabel("Observation Index")
axes[1].legend()

axes[2].scatter(leverage, residuals / residuals.std(), alpha=0.6)
axes[2].set_xlabel("Leverage")
axes[2].set_ylabel("Studentized Residuals")
axes[2].set_title("Influence Plot")
plt.tight_layout()
plt.show()
```

#### 5.5.6 Confidence Intervals for Predictions

Linear models naturally provide prediction intervals — a key advantage:

```python
# Prediction intervals using statsmodels
predictions = sm_model.get_prediction(sm.add_constant(X_test))
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI

pred_summary.head()

# Plot with confidence intervals
fig, ax = plt.subplots(figsize=(14, 5))
test_dates = df.index[split_idx:]

ax.plot(test_dates, y_test, label="Actual", alpha=0.7)
ax.plot(test_dates, pred_summary["mean"], label="Predicted", alpha=0.7)
ax.fill_between(test_dates, 
                 pred_summary["obs_ci_lower"],
                 pred_summary["obs_ci_upper"],
                 alpha=0.2, label="95% Prediction Interval")
ax.set_title("Predictions with 95% Confidence Intervals")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price ($)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Coverage: what % of actuals fall within the prediction interval?
within_ci = ((y_test >= pred_summary["obs_ci_lower"].values) & 
             (y_test <= pred_summary["obs_ci_upper"].values)).mean()
print(f"Actual values within 95% prediction interval: {within_ci:.1%}")
```

**Evaluation checklist for linear models:**

| Check | Method | Target |
|---|---|---|
| Feature significance | p-value from OLS summary | p < 0.05 |
| Multicollinearity | VIF | VIF < 5 |
| Homoscedasticity | Breusch-Pagan / White test | p > 0.05 |
| Residual normality | D'Agostino-Pearson / Q-Q plot | p > 0.05 |
| Autocorrelation | Durbin-Watson / Ljung-Box | DW ≈ 2.0 |
| Influential points | Cook's distance | Max < 4/n |
| Prediction coverage | CI empirical coverage | ~95% |

### 5.6 In-Depth Evaluation for Cluster-Based Models

Cluster models (K-Means, DBSCAN, Hierarchical, Gaussian Mixture Models) are **unsupervised** — there is no ground truth label to compare against. Evaluation focuses on internal coherence, separation, and stability.

#### 5.6.1 Internal Validation Metrics

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Prepare clustering data (e.g., stock returns)
returns_data = df["XOM_Stock"].pct_change().dropna().values.reshape(-1, 1)

# Try different k values
k_range = range(2, 11)
sil_scores = []
db_scores = []
ch_scores = []
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(returns_data)
    
    sil_scores.append(silhouette_score(returns_data, labels))
    db_scores.append(davies_bouldin_score(returns_data, labels))
    ch_scores.append(calinski_harabasz_score(returns_data, labels))
    inertias.append(kmeans.inertia_)

# Plot all metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(k_range, sil_scores, "bo-")
axes[0, 0].axvline(x=np.argmax(sil_scores) + 2, color="red", linestyle="--", label=f"Best k={np.argmax(sil_scores)+2}")
axes[0, 0].set_title("Silhouette Score (↑ better)")
axes[0, 0].set_xlabel("Number of clusters (k)")
axes[0, 0].legend()

axes[0, 1].plot(k_range, db_scores, "ro-")
axes[0, 1].axvline(x=np.argmin(db_scores) + 2, color="red", linestyle="--", label=f"Best k={np.argmin(db_scores)+2}")
axes[0, 1].set_title("Davies-Bouldin Index (↓ better)")
axes[0, 1].set_xlabel("Number of clusters (k)")
axes[0, 1].legend()

axes[1, 0].plot(k_range, ch_scores, "go-")
axes[1, 0].axvline(x=np.argmax(ch_scores) + 2, color="red", linestyle="--", label=f"Best k={np.argmax(ch_scores)+2}")
axes[1, 0].set_title("Calinski-Harabasz Index (↑ better)")
axes[1, 0].set_xlabel("Number of clusters (k)")
axes[1, 0].legend()

axes[1, 1].plot(k_range, inertias, "mo-")
axes[1, 1].axvline(x=np.argmax(np.diff(inertias, 2)) + 2, color="red", linestyle="--", label="Elbow point")
axes[1, 1].set_title("Inertia / Elbow Method")
axes[1, 1].set_xlabel("Number of clusters (k)")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Summary table
eval_df = pd.DataFrame({
    "k": k_range,
    "Silhouette": sil_scores,
    "Davies-Bouldin": db_scores,
    "Calinski-Harabasz": ch_scores,
    "Inertia": inertias
})
print(eval_df.round(4))
```

**Cluster metric interpretation:**

| Metric | Range | Ideal | Key Property |
|---|---|---|---|
| **Silhouette Score** | [-1, 1] | Max | Measures cohesion (intra-cluster) vs. separation (inter-cluster). > 0.5 = reasonable, > 0.7 = strong |
| **Davies-Bouldin Index** | [0, ∞) | Min | Average similarity between each cluster and its most similar neighbour. Lower = better separated |
| **Calinski-Harabasz Index** | [0, ∞) | Max | Ratio of between-cluster variance to within-cluster variance. Higher = more compact + separated |
| **Inertia** | [0, ∞) | Elbow | Sum of squared distances to centroids. Always decreases with k; look for the "elbow" |

#### 5.6.2 Gap Statistic for Optimal k

The gap statistic compares the within-cluster dispersion to a null reference distribution:

```python
def gap_statistic(X, k_max=10, n_references=10):
    """
    Calculate Gap statistic: measures how much better clustering is 
    vs. uniform random data.
    """
    n = len(X)
    gaps = []
    gap_stds = []
    
    for k in range(1, k_max + 1):
        # Fit on real data
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        real_inertia = kmeans.inertia_
        log_real = np.log(real_inertia)
        
        # Fit on reference (uniform random) data
        ref_inertias = []
        for _ in range(n_references):
            X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=5)
            kmeans_ref.fit(X_ref)
            ref_inertias.append(np.log(kmeans_ref.inertia_))
        
        gap = np.mean(ref_inertias) - log_real
        gaps.append(gap)
        gap_stds.append(np.std(ref_inertias) * np.sqrt(1 + 1/n_references))
    
    return gaps, gap_stds

gaps, gap_stds = gap_statistic(returns_data, k_max=10)

# Plot gap statistic
k_range = range(1, 11)
fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(k_range, gaps, yerr=gap_stds, fmt="bo-", capsize=5)
ax.set_title("Gap Statistic (↑ better; optimal k where gap plateaus)")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Gap statistic")
ax.axhline(y=0, color="gray", linestyle="--")
plt.tight_layout()
plt.show()
```

#### 5.6.3 Cluster Stability Analysis

A good clustering solution should be stable under data perturbations:

```python
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def cluster_stability_analysis(X, n_clusters=3, n_bootstrap=100):
    """
    Assess cluster stability by comparing clusterings on bootstrapped samples.
    High stability → consistent ARI/NMI scores across resamples.
    """
    # Reference clustering on full data
    ref_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    ref_labels = ref_kmeans.fit_predict(X)
    
    ari_scores = []
    nmi_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        X_boot = resample(X, replace=True, n_samples=len(X))
        
        # Cluster on bootstrap
        boot_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        boot_labels = boot_kmeans.fit_predict(X_boot)
        
        # Align labels by matching reference clustering on overlapping indices
        # (ARI handles label alignment automatically)
        ari = adjusted_rand_score(ref_labels, boot_labels[:len(ref_labels)] 
                                   if len(boot_labels) >= len(ref_labels) 
                                   else np.append(boot_labels, [-1] * (len(ref_labels) - len(boot_labels))))
        nmi = normalized_mutual_info_score(ref_labels, boot_labels[:len(ref_labels)]
                                            if len(boot_labels) >= len(ref_labels)
                                            else np.append(boot_labels, [-1] * (len(ref_labels) - len(boot_labels))))
        ari_scores.append(ari)
        nmi_scores.append(nmi)
    
    print(f"Cluster Stability (k={n_clusters}, {n_bootstrap} bootstraps):")
    print(f"  Adjusted Rand Index:       {np.mean(ari_scores):.4f} ± {np.std(ari_scores):.4f}")
    print(f"  Normalized Mutual Info:    {np.mean(nmi_scores):.4f} ± {np.std(nmi_scores):.4f}")
    print(f"\nInterpretation:")
    print(f"  Mean ARI > 0.9:  Very stable (excellent)")
    print(f"  Mean ARI 0.7-0.9: Stable (good)")
    print(f"  Mean ARI 0.5-0.7: Moderate (acceptable)")
    print(f"  Mean ARI < 0.5:  Unstable (re-evaluate features or k)")
    
    return ari_scores, nmi_scores

ari, nmi = cluster_stability_analysis(returns_data, n_clusters=3)
```

#### 5.6.4 Visual Evaluation and Cluster Profiles

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Fit final clustering
final_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(returns_data)

# PCA projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(returns_data)

# t-SNE projection
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(returns_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA plot
for label in np.unique(cluster_labels):
    mask = cluster_labels == label
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {label}", alpha=0.7)
axes[0].scatter(final_kmeans.cluster_centers_[:, 0], [0]*3, 
                marker="x", s=200, color="black", label="Centroids")
axes[0].set_title(f"PCA Projection (explained var: {pca.explained_variance_ratio_.sum():.2%})")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()

# t-SNE plot
for label in np.unique(cluster_labels):
    mask = cluster_labels == label
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f"Cluster {label}", alpha=0.7)
axes[1].set_title("t-SNE Projection")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")
axes[1].legend()

plt.tight_layout()
plt.show()

# Cluster profiles
cluster_profiles = pd.DataFrame({
    "Cluster": cluster_labels,
    "Return": returns_data.flatten()
})
print("Cluster Profiles:")
print(cluster_profiles.groupby("Cluster").describe().round(4))
```

#### 5.6.5 External Validation (When Labels Exist)

If partial labels exist (e.g., known market regimes: bull, bear, sideways):

```python
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

# Simulated ground truth labels (e.g., market regimes)
def simulate_regime_labels(returns, threshold_high=0.01, threshold_low=-0.01):
    """Simulate market regime labels based on return thresholds."""
    rolling_mean = pd.Series(returns.flatten()).rolling(21).mean().fillna(0)
    labels = np.zeros(len(rolling_mean))
    labels[rolling_mean > threshold_high] = 1   # bull
    labels[rolling_mean < threshold_low] = 2    # bear
    return labels.astype(int)

true_labels = simulate_regime_labels(returns_data)

# Compare clustering vs. true regimes
print("External Cluster Validation:")
print(f"  Adjusted Rand Index:     {adjusted_rand_score(true_labels, cluster_labels):.4f}")
print(f"  Homogeneity:             {homogeneity_score(true_labels, cluster_labels):.4f}")
print(f"  Completeness:            {completeness_score(true_labels, cluster_labels):.4f}")
print(f"  V-Measure:               {v_measure_score(true_labels, cluster_labels):.4f}")
```

**Evaluation checklist for cluster-based models:**

| Check | Method | Target |
|---|---|---|
| Optimal k | Gap statistic + Silhouette + Elbow | Consensus across methods |
| Cluster separation | Davies-Bouldin, CH Index | Minimise DB, maximise CH |
| Cluster cohesion | Silhouette Score | > 0.5 |
| Stability | Bootstrap ARI / NMI | ARI > 0.7 |
| Visual sanity | PCA / t-SNE projection | Non-overlapping clusters |
| Interpretability | Cluster profiles (mean, std per feature) | Meaningful cluster differences |

### 5.7 Comparing Linear vs. Cluster Model Evaluation

| Aspect | Linear Models | Cluster Models |
|---|---|---|
| **Supervision** | Supervised (y available) | Unsupervised (no y) |
| **Primary metrics** | R², RMSE, MAE, MAPE | Silhouette, DB Index, CH Index |
| **Assumptions** | Linearity, normality, homoscedasticity | Distance metric dependency |
| **Validation** | Hold-out set, walk-forward | Stability analysis, gap statistic |
| **Interpretability** | Coefficients, p-values, CI | Centroid profiles, projections |
| **Drift sensitivity** | Residual distribution, coefficient stability | Cluster assignment distribution |

---

## 6. Model Optimisation Techniques

Once a baseline model is established, we optimise to improve performance.

### 6.1 Hyperparameter Tuning

#### Grid Search for Ridge Regression

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "solver": ["auto", "svd", "cholesky", "lbfgs"]
}

ridge = Ridge()
grid_search = GridSearchCV(
    ridge, param_grid, cv=tscv, 
    scoring="r2", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

best_ridge = grid_search.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)
print(f"Test R² with optimised Ridge: {r2_score(y_test, y_pred_ridge):.4f}")
```

#### Random Search for Gradient Boosting

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(3, 15),
    "learning_rate": uniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4),
    "min_samples_split": randint(2, 20)
}

gbr = GradientBoostingRegressor(random_state=42)
random_search = RandomizedSearchCV(
    gbr, param_dist, n_iter=50, cv=tscv,
    scoring="r2", n_jobs=-1, random_state=42, verbose=1
)
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV R²: {random_search.best_score_:.4f}")
```

### 6.2 Regularisation

Regularisation prevents overfitting by penalising large coefficients.

```python
# Compare Linear, Ridge, Lasso
models_reg = {
    "Linear (no regularisation)": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.01),
    "ElasticNet (L1+L2)": ElasticNet(alpha=0.01, l1_ratio=0.5)
}

fig, ax = plt.subplots(figsize=(12, 6))

for name, model in models_reg.items():
    model.fit(X_train, y_train)
    coefs = model.coef_ if hasattr(model, "coef_") else [0]
    label = f"{name} (|coef| sum: {np.sum(np.abs(coefs)):.2f})"
    
    # For visualisation: use first 20 features if many
    if len(coefs) > 20:
        coefs = coefs[:20]
    
    ax.plot(coefs, marker="o", label=label)

ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Coefficient Comparison: Regularisation Effects")
ax.set_xlabel("Feature Index")
ax.set_ylabel("Coefficient Value")
ax.legend()
plt.tight_layout()
plt.show()
```

### 6.3 Feature Scaling

For many algorithms (especially regularised regression and SVR), feature scaling is essential:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare performance with and without scaling
models_scale = {
    "Linear (no scaling)": (LinearRegression(), X_train, X_test),
    "Linear (StandardScaler)": (LinearRegression(), X_train_scaled, X_test_scaled),
    "SVR (no scaling)": (SVR(kernel="rbf"), X_train, X_test),
    "SVR (StandardScaler)": (SVR(kernel="rbf"), X_train_scaled, X_test_scaled),
}

for name, (model, X_tr, X_te) in models_scale.items():
    model.fit(X_tr, y_train)
    y_p = model.predict(X_te)
    r2 = r2_score(y_test, y_p)
    print(f"  {name}: R² = {r2:.4f}")
```

### 6.4 Ensemble Methods

Combining multiple models often yields better performance than any single model:

```python
from sklearn.ensemble import VotingRegressor, StackingRegressor

# Voting ensemble (average of predictions)
voting = VotingRegressor([
    ("lr", LinearRegression()),
    ("ridge", Ridge(alpha=1.0)),
    ("gbr", GradientBoostingRegressor(n_estimators=100, max_depth=3))
])
voting.fit(X_train, y_train)
y_pred_voting = voting.predict(X_test)

print(f"Voting Ensemble R²: {r2_score(y_test, y_pred_voting):.4f}")

# Stacking ensemble (meta-model learns from base models)
base_models = [
    ("lr", LinearRegression()),
    ("ridge", Ridge(alpha=1.0)),
    ("rf", RandomForestRegressor(n_estimators=100, max_depth=5))
]
meta_model = Ridge(alpha=1.0)

stacking = StackingRegressor(base_models, meta_model, cv=5)
stacking.fit(X_train, y_train)
y_pred_stacking = stacking.predict(X_test)

print(f"Stacking Ensemble R²: {r2_score(y_test, y_pred_stacking):.4f}")
```

### 6.5 Optimisation Insights

| Optimisation Goal | Technique | Impact |
|---|---|---|
| **Speed** | Linear Regression (closed-form solution) | ~1ms inference |
| **Stability** | Ridge regularisation | Robust to collinear features |
| **Feature Selection** | Lasso (L1) | Drives irrelevant coefficients to zero |
| **Generalisability** | Cross-validation, regularisation | Prevents overfitting |

---

## 7. Data & Concept Drift (Evidently + SHAP)

This section is the heart of production ML monitoring. Models degrade over time as the underlying data distribution changes — detecting this drift is critical for maintaining prediction quality. The approach differs substantially between **linear models** (parametric, assumption-driven) and **cluster-based models** (unsupervised, pattern-driven).

### 7.1 What is Drift?

**Data Drift (Covariate Shift):** The distribution of input features changes over time.

```
P_train(features) ≠ P_production(features)
```

**Concept Drift:** The relationship between features and the target changes.

```
P_train(target | features) ≠ P_production(target | features)
```

**Prediction Drift:** The distribution of model outputs changes.

```
P_train(predictions) ≠ P_production(predictions)
```

Visual analogy:

```
Training Data Distribution          Current Production Distribution
       ╭───╮                                ╭──────╮
      ╱     ╲                              ╱        ╲
     ╱       ╲               →            ╱          ╲
    ╱         ╲                          ╱            ╲
   ╰───────────╯                        ╰──────────────╯
        (shifted right, wider)
```

### 7.2 Statistical Drift Detection Methods

#### Population Stability Index (PSI)

PSI measures how much a distribution has shifted. Values > 0.25 indicate significant drift.

```python
def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index.
    PSI = Σ (p_i - q_i) × ln(p_i / q_i)
    where p_i = proportion in bin i for expected, q_i = proportion for actual
    """
    # Create bins based on expected distribution
    breaks = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    
    # Bin both distributions
    expected_bins = np.clip(np.digitize(expected, breaks) - 1, 0, bins)
    actual_bins = np.clip(np.digitize(actual, breaks) - 1, 0, bins)
    
    psi = 0
    for i in range(bins):
        p_i = np.mean(expected_bins == i) + 1e-10  # avoid log(0)
        q_i = np.mean(actual_bins == i) + 1e-10
        psi += (p_i - q_i) * np.log(p_i / q_i)
    
    return psi

# Simulate drift: training oil prices vs. recent oil prices
np.random.seed(42)
train_oil = np.random.normal(60, 10, 1000)     # training distribution
prod_oil  = np.random.normal(75, 15, 100)       # production distribution (drifted)

psi_value = calculate_psi(train_oil, prod_oil)
print(f"PSI: {psi_value:.4f}")
if psi_value < 0.1:
    print("✓ No significant drift (PSI < 0.1)")
elif psi_value < 0.25:
    print("⚠ Moderate drift detected (0.1 ≤ PSI < 0.25) — investigate")
else:
    print("✗ Significant drift (PSI ≥ 0.25) — retrain required")
```

#### Kullback-Leibler (KL) Divergence

```python
from scipy.special import rel_entr

def kl_divergence(p, q, bins=50):
    """Calculate KL Divergence between two distributions."""
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    
    # Add small epsilon to avoid log(0)
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    
    kl_div = np.sum(rel_entr(p_hist, q_hist))
    return kl_div

kl = kl_divergence(train_oil, prod_oil)
print(f"KL Divergence: {kl:.4f}")
# KL = 0 means identical distributions; higher values indicate more drift
```

#### Kolmogorov-Smirnov (KS) Test

```python
from scipy.stats import ks_2samp

ks_stat, ks_p = ks_2samp(train_oil, prod_oil)
print(f"KS Statistic: {ks_stat:.4f}, p-value: {ks_p:.4e}")
if ks_p > 0.05:
    print("✓ No statistically significant drift (p > 0.05)")
else:
    print("✗ Statistically significant drift detected (p ≤ 0.05)")
```

### 7.3 Drift Detection for Linear Models

Linear models are parametric: they assume a specific functional form `y = Xβ + ε`. Drift detection focuses on whether the parameters (coefficients) and error structure remain stable over time.

#### 7.3.1 Coefficient Stability (Rolling Regression)

Monitor if the model coefficients change over time — the earliest warning sign of concept drift:

```python
def rolling_coefficient_stability(X, y, window=252, model_class=LinearRegression):
    """
    Fit the model on rolling windows and track coefficient trajectories.
    Large changes in coefficients = concept drift in the underlying relationship.
    """
    n = len(X)
    coefs = []
    intercepts = []
    r2_scores = []
    dates = []
    
    for end in range(window, n + 1):
        X_window = X[end - window:end]
        y_window = y[end - window:end]
        
        model = model_class()
        model.fit(X_window, y_window)
        
        coefs.append(model.coef_[0])
        intercepts.append(model.intercept_)
        r2_scores.append(model.score(X_window, y_window))
    
    # Plot coefficient trajectory
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    ax1 = axes[0]
    ax1.plot(coefs, label="β₁ (oil price coefficient)", color="blue")
    ax1.axhline(y=np.mean(coefs), color="blue", linestyle="--", alpha=0.5, label=f"Mean β₁ = {np.mean(coefs):.4f}")
    ax1.fill_between(range(len(coefs)), 
                      np.mean(coefs) - 2*np.std(coefs), 
                      np.mean(coefs) + 2*np.std(coefs), 
                      alpha=0.2, label="±2σ band")
    ax1.set_title("Rolling Coefficient Stability (Linear Model)")
    ax1.set_ylabel("Coefficient Value (β₁)")
    ax1.legend()
    
    ax2 = axes[1]
    ax2.plot(r2_scores, label="Rolling R²", color="green")
    ax2.axhline(y=np.mean(r2_scores), color="green", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Rolling Window Index")
    ax2.set_ylabel("R² Score")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Drift signal: coefficient outside ±2σ from stable mean
    current_coef = coefs[-1]
    mean_coef = np.mean(coefs)
    std_coef = np.std(coefs)
    
    print(f"Current β₁ = {current_coef:.4f}")
    print(f"Reference β₁ = {mean_coef:.4f} ± {std_coef:.4f}")
    if abs(current_coef - mean_coef) > 2 * std_coef:
        print("⚠ ALERT: Coefficient drift detected (beyond 2σ)")
    else:
        print("✓ Coefficient stable within 2σ")
    
    return coefs, intercepts, r2_scores

# Run on our data
coefs, intercepts, r2_scores = rolling_coefficient_stability(X, y, window=252)
```

#### 7.3.2 Residual Monitoring for Linear Models

Linear model residuals have a known expected distribution (N(0, σ²)). Deviations signal drift:

```python
def residual_drift_monitoring(model, X_ref, y_ref, X_current, y_current, window=63):
    """Monitor residual mean and variance shifts for linear models."""
    
    # Reference residuals
    y_pred_ref = model.predict(X_ref)
    residuals_ref = y_ref - y_pred_ref
    
    # Current residuals
    y_pred_curr = model.predict(X_current)
    residuals_curr = y_current - y_pred_curr
    
    # Test mean shift (t-test)
    t_stat, t_p = stats.ttest_ind(residuals_ref, residuals_curr)
    
    # Test variance shift (F-test for variance ratio)
    var_ratio = np.var(residuals_ref) / np.var(residuals_curr)
    f_p = 2 * min(stats.f.cdf(var_ratio, len(residuals_ref)-1, len(residuals_curr)-1),
                  1 - stats.f.cdf(var_ratio, len(residuals_ref)-1, len(residuals_curr)-1))
    
    # Cumulative tracking
    residual_means = []
    residual_stds = []
    
    for i in range(0, len(y_current), window):
        chunk_resid = residuals_curr[i:i+window]
        if len(chunk_resid) > 0:
            residual_means.append(chunk_resid.mean())
            residual_stds.append(chunk_resid.std())
    
    print("--- Linear Model Residual Drift ---")
    print(f"Reference residual mean: {residuals_ref.mean():.4f} ± {residuals_ref.std():.4f}")
    print(f"Current residual mean:  {residuals_curr.mean():.4f} ± {residuals_curr.std():.4f}")
    print(f"\nMean shift t-test:   t={t_stat:.4f}, p={t_p:.4f}")
    print(f"Variance ratio F-test: p={f_p:.4f}")
    
    if t_p < 0.05:
        print("⚠ ALERT: Residual mean shifted (biased predictions)")
    else:
        print("✓ Residual mean stable")
    
    if f_p < 0.05:
        print("⚠ ALERT: Residual variance changed (uncertainty shift)")
    else:
        print("✓ Residual variance stable")
```

#### 7.3.3 CUSUM for Structural Break Detection

CUSUM (Cumulative Sum) detects subtle, sustained shifts in model errors:

```python
def cusum_detection(residuals, threshold=5):
    """
    CUSUM algorithm for detecting structural breaks in prediction errors.
    Accumulates deviations from zero mean; crossing threshold = break detected.
    """
    # Standardise residuals
    std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    
    # Cumulative sum
    cumsum = np.cumsum(std_residuals)
    
    # Upper and lower CUSUM
    pos_cusum = np.zeros(len(cumsum))
    neg_cusum = np.zeros(len(cumsum))
    
    for i in range(1, len(cumsum)):
        pos_cusum[i] = max(0, cumsum[i] - threshold)
        neg_cusum[i] = abs(min(0, cumsum[i] + threshold))
    
    # Detection: where CUSUM exceeds threshold
    change_points = np.where((pos_cusum > 0) | (neg_cusum > 0))[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    
    axes[0].plot(std_residuals, alpha=0.6)
    axes[0].axhline(y=0, color="gray", linestyle="--")
    axes[0].set_title("Standardised Residuals (CUSUM Analysis)")
    axes[0].set_ylabel("Standardised Residual")
    
    axes[1].plot(cumsum, label="Cumulative Sum")
    axes[1].axhline(y=threshold, color="red", linestyle="--", label=f"+{threshold}σ threshold")
    axes[1].axhline(y=-threshold, color="red", linestyle="--")
    if len(change_points) > 0:
        axes[1].axvline(x=change_points[0], color="orange", linestyle=":", 
                        label=f"Change point at t={change_points[0]}")
    axes[1].fill_between(range(len(cumsum)), threshold, cumsum, 
                         where=(cumsum > threshold), color="red", alpha=0.3)
    axes[1].fill_between(range(len(cumsum)), -threshold, cumsum, 
                         where=(cumsum < -threshold), color="red", alpha=0.3)
    axes[1].set_title("CUSUM Structural Break Detection")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Cumulative Sum")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    if len(change_points) > 0:
        print(f"⚠ ALERT: Structural break detected at observation {change_points[0]}")
    else:
        print("✓ No structural break detected")
    
    return change_points

# Use test set residuals
cusum_detection(residuals)
```

#### 7.3.4 Chow Test for Parameter Stability

The Chow test explicitly tests whether coefficients are equal across two periods:

```python
def chow_test(X, y, split_idx):
    """
    Chow test: tests whether regression coefficients differ between two sub-samples.
    H₀: coefficients are equal (no structural break)
    H₁: coefficients differ (structural break at split_idx)
    """
    # Full model
    X_full = sm.add_constant(X)
    model_full = sm.OLS(y, X_full).fit()
    rss_full = model_full.ssr
    
    # Sub-model 1 (before split)
    X1 = sm.add_constant(X[:split_idx])
    y1 = y[:split_idx]
    model1 = sm.OLS(y1, X1).fit()
    rss1 = model1.ssr
    
    # Sub-model 2 (after split)
    X2 = sm.add_constant(X[split_idx:])
    y2 = y[split_idx:]
    model2 = sm.OLS(y2, X2).fit()
    rss2 = model2.ssr
    
    # Chow statistic
    n = len(y)
    k = X_full.shape[1]
    chow_stat = ((rss_full - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n - 2*k))
    chow_p = 1 - stats.f.cdf(chow_stat, k, n - 2*k)
    
    print("--- Chow Test for Parameter Stability ---")
    print(f"F-statistic: {chow_stat:.4f}")
    print(f"p-value:     {chow_p:.4f}")
    if chow_p < 0.05:
        print("⚠ ALERT: Structural break detected (coefficients differ between periods)")
    else:
        print("✓ Coefficients stable across periods")
    
    return chow_stat, chow_p

# Test if relationship changed in the last 20% of data
chow_stat, chow_p = chow_test(X, y, split_idx)
```

#### 7.3.5 Leverage and Influence Monitoring

Track whether new production data points are becoming "influential" — far from the training distribution:

```python
def monitor_leverage_drift(X_train, X_current, threshold_percentile=95):
    """
    Monitor the leverage (hat values) of incoming data points.
    High leverage = far from training distribution → prediction is extrapolation.
    """
    from scipy.spatial.distance import mahalanobis
    
    # Training distribution statistics
    train_mean = X_train.mean(axis=0)
    train_cov = np.cov(X_train.T)
    
    # Mahalanobis distance for each new point
    distances = []
    for x in X_current:
        d = mahalanobis(x.flatten(), train_mean, np.linalg.inv(train_cov))
        distances.append(d)
    
    # Threshold from training data
    train_distances = [mahalanobis(x, train_mean, np.linalg.inv(train_cov)) 
                       for x in X_train]
    threshold = np.percentile(train_distances, threshold_percentile)
    
    # Fraction of high-leverage points in current data
    high_leverage = np.sum(np.array(distances) > threshold)
    fraction = high_leverage / len(X_current)
    
    print("--- Leverage Drift Monitoring ---")
    print(f"Training 95th percentile Mahalanobis distance: {threshold:.2f}")
    print(f"Current data points exceeding threshold: {high_leverage}/{len(X_current)} ({fraction:.1%})")
    
    if fraction > 0.1:
        print(f"⚠ ALERT: {fraction:.1%} of current data are extrapolations (high leverage)")
    elif fraction > 0.05:
        print(f"⚠ CAUTION: {fraction:.1%} of current data are extrapolations")
    else:
        print(f"✓ Most predictions are interpolations (within training range)")
    
    return np.array(distances), threshold
```

### 7.4 Drift Detection for Cluster-Based Models

Cluster models group data based on similarity. Drift manifests as changes in cluster structure rather than parameter shifts.

#### 7.4.1 Cluster Assignment Stability Over Time

Track whether data points are being assigned to the same clusters as before:

```python
def cluster_assignment_drift(cluster_model, X_reference, X_current):
    """
    Monitor changes in cluster assignment distributions.
    A significant shift in which clusters are populated = distribution drift.
    """
    # Reference assignments
    ref_labels = cluster_model.predict(X_reference)
    ref_distribution = pd.Series(ref_labels).value_counts(normalize=True).sort_index()
    
    # Current assignments
    curr_labels = cluster_model.predict(X_current)
    curr_distribution = pd.Series(curr_labels).value_counts(normalize=True).sort_index()
    
    # Compare distributions
    all_clusters = sorted(set(ref_distribution.index) | set(curr_distribution.index))
    
    print("--- Cluster Assignment Drift ---")
    print(f"{'Cluster':<10} {'Reference %':<15} {'Current %':<15} {'Change':<15}")
    print("-" * 55)
    
    drift_score = 0
    for c in all_clusters:
        ref_pct = ref_distribution.get(c, 0) * 100
        curr_pct = curr_distribution.get(c, 0) * 100
        change = curr_pct - ref_pct
        drift_score += abs(change) / 100
        print(f"{c:<10} {ref_pct:<15.2f} {curr_pct:<15.2f} {change:<+15.2f}")
    
    print(f"\nTotal Drift Score (sum of |Δ%|): {drift_score:.2%}")
    if drift_score > 0.3:
        print("⚠ ALERT: Significant cluster redistribution detected")
    elif drift_score > 0.15:
        print("⚠ CAUTION: Moderate cluster redistribution")
    else:
        print("✓ Cluster assignments stable")
    
    # Visualise
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(all_clusters))
    width = 0.35
    ax.bar(x - width/2, [ref_distribution.get(c, 0)*100 for c in all_clusters], 
           width, label="Reference", alpha=0.8)
    ax.bar(x + width/2, [curr_distribution.get(c, 0)*100 for c in all_clusters], 
           width, label="Current", alpha=0.8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Cluster Assignment Distribution: Reference vs. Current")
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return drift_score

# Fit a KMeans model for monitoring
cluster_model = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_model.fit(X_train)

# Check assignment drift on test set
cluster_drift = cluster_assignment_drift(cluster_model, X_train, X_test)
```

#### 7.4.2 Centroid Drift Tracking

Monitor whether cluster centroids are shifting in feature space:

```python
def centroid_drift_tracking(initial_model, X_current, n_clusters=3):
    """
    Track how much current data centroids drift from the initial centroids.
    Uses the current data's nearest-centroid assignments.
    """
    # Initial centroids (from training)
    initial_centroids = initial_model.cluster_centers_
    
    # Re-fit on current data to get current centroids
    current_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    current_model.fit(X_current)
    current_centroids = current_model.cluster_centers_
    
    # Align centroids by matching (minimise total Euclidean distance)
    from scipy.optimize import linear_sum_assignment
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = np.linalg.norm(initial_centroids[i] - current_centroids[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    print("--- Centroid Drift Tracking ---")
    total_drift = 0
    for i, j in zip(row_ind, col_ind):
        drift = cost_matrix[i, j]
        total_drift += drift
        print(f"Cluster {i}: centroid drift = {drift:.4f}")
    
    avg_drift = total_drift / n_clusters
    print(f"\nAverage centroid drift: {avg_drift:.4f}")
    
    # Scale-dependent: compare to initial cluster spread
    initial_spread = np.mean([np.std(X_current, axis=0).mean()])
    relative_drift = avg_drift / initial_spread
    print(f"Relative drift (vs. feature std): {relative_drift:.2%}")
    
    if relative_drift > 0.5:
        print("⚠ ALERT: Large centroid shift detected")
    elif relative_drift > 0.25:
        print("⚠ CAUTION: Moderate centroid shift")
    else:
        print("✓ Centroids stable")
    
    return avg_drift
```

#### 7.4.3 Silhouette Width Degradation

Monitor whether cluster quality is degrading over time:

```python
def silhouette_monitoring(cluster_model, X_reference, X_current, window=63):
    """
    Track silhouette score over time. A sustained decrease signals that
    the learned cluster structure no longer fits the data well.
    """
    # Reference silhouette
    ref_labels = cluster_model.predict(X_reference)
    ref_sil = silhouette_score(X_reference, ref_labels)
    
    # Rolling silhouette on current data
    sil_scores = []
    for i in range(0, len(X_current) - window + 1, window):
        X_chunk = X_current[i:i+window]
        chunk_labels = cluster_model.predict(X_chunk)
        if len(np.unique(chunk_labels)) > 1:  # silhouette requires at least 2 clusters
            sil = silhouette_score(X_chunk, chunk_labels)
            sil_scores.append(sil)
    
    print("--- Silhouette Degradation Monitoring ---")
    print(f"Reference Silhouette: {ref_sil:.4f}")
    print(f"Current Silhouette:   {np.mean(sil_scores):.4f} (min: {np.min(sil_scores):.4f})")
    
    degradation = ref_sil - np.mean(sil_scores)
    print(f"Degradation: {degradation:.4f}")
    
    if degradation > 0.2:
        print("⚠ ALERT: Severe cluster quality degradation")
    elif degradation > 0.1:
        print("⚠ CAUTION: Moderate cluster quality degradation")
    else:
        print("✓ Cluster quality maintained")
    
    # Plot trajectory
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sil_scores, "bo-", label="Rolling Silhouette")
    ax.axhline(y=ref_sil, color="green", linestyle="--", label=f"Reference ({ref_sil:.3f})")
    ax.axhline(y=ref_sil - 0.1, color="orange", linestyle=":", label="Caution threshold")
    ax.axhline(y=ref_sil - 0.2, color="red", linestyle=":", label="Alert threshold")
    ax.set_title("Silhouette Score Over Time")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
    plt.tight_layout()
    plt.show()
```

#### 7.4.4 Cluster Feature Distribution Shift

For each cluster, monitor whether the within-cluster feature distribution is shifting:

```python
def per_cluster_feature_drift(cluster_model, X_reference, X_current, feature_names):
    """
    Monitor feature distribution drift within each cluster separately.
    A cluster whose features are drifting internally may be splitting or morphing.
    """
    ref_labels = cluster_model.predict(X_reference)
    curr_labels = cluster_model.predict(X_current)
    
    n_clusters = len(np.unique(ref_labels))
    results = []
    
    for c in range(n_clusters):
        ref_cluster_data = X_reference[ref_labels == c]
        curr_cluster_data = X_current[curr_labels == c]
        
        if len(ref_cluster_data) == 0 or len(curr_cluster_data) == 0:
            continue
        
        for f_idx, f_name in enumerate(feature_names):
            ref_feat = ref_cluster_data[:, f_idx]
            curr_feat = curr_cluster_data[:, f_idx]
            
            # PSI for this feature in this cluster
            psi = calculate_psi(ref_feat, curr_feat)
            
            # KS test
            ks_stat, ks_p = ks_2samp(ref_feat, curr_feat)
            
            results.append({
                "Cluster": c,
                "Feature": f_name,
                "PSI": psi,
                "KS_p": ks_p,
                "Drift": "YES" if psi > 0.25 or ks_p < 0.05 else "no"
            })
    
    results_df = pd.DataFrame(results)
    print("--- Per-Cluster Feature Drift ---")
    print(results_df.to_string(index=False))
    
    # Summary
    drifted = results_df[results_df["Drift"] == "YES"]
    print(f"\nDrifted clusters/features: {len(drifted)} / {len(results_df)}")
    if len(drifted) > 0:
        print("⚠ ALERT: Some cluster-feature combinations show drift")
        for _, row in drifted.iterrows():
            print(f"  Cluster {row['Cluster']}, Feature {row['Feature']}: PSI={row['PSI']:.3f}")
    else:
        print("✓ All cluster-feature distributions stable")
    
    return results_df
```

### 7.5 Evidently AI — Production-Grade Drift Monitoring

[Evidently AI](https://www.evidentlyai.com/) is an open-source library for monitoring ML models in production. It provides pre-built reports and dashboards for data drift, target drift, regression performance, and more.

#### Installation

```bash
pip install evidently
```

#### Data Drift Report

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift

# Create reference (training) and current (production) datasets
reference_data = pd.DataFrame({
    "WTI_Oil": np.random.normal(60, 10, 1000),
    "XOM_predicted": np.random.normal(75, 5, 1000)
})

current_data = pd.DataFrame({
    "WTI_Oil": np.random.normal(75, 15, 100),   # drifted distribution
    "XOM_predicted": np.random.normal(82, 8, 100)
})

# Create Evidently Data Drift Report
data_drift_report = Report(metrics=[
    DataDriftPreset()
])

data_drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=None
)

# Display the report
data_drift_report

# Save as HTML for sharing
data_drift_report.save_html("data_drift_report.html")
print("Data drift report saved to data_drift_report.html")
```

#### Regression Model Performance Report

```python
reg_performance_report = Report(metrics=[
    RegressionPreset()
])

# Simulate reference predictions
ref_predictions = 0.8 * reference_data["WTI_Oil"] + 25 + np.random.normal(0, 3, 1000)

# Simulate current predictions (with drift)
curr_predictions = 0.5 * current_data["WTI_Oil"] + 35 + np.random.normal(0, 5, 100)

reference_data["XOM_actual"] = ref_predictions
current_data["XOM_actual"] = curr_predictions

reg_performance_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping={
        "target": "XOM_actual",
        "prediction": "XOM_predicted",
        "numerical_features": ["WTI_Oil"]
    }
)

reg_performance_report.show()

reg_performance_report.save_html("regression_performance_report.html")
print("Regression performance report saved to regression_performance_report.html")
```

#### Evidently Test Suite (for Automated Alerts)

```python
drift_test_suite = TestSuite(tests=[
    TestColumnDrift(column_name="WTI_Oil"),
    TestColumnDrift(column_name="XOM_predicted"),
])

drift_test_suite.run(
    reference_data=reference_data,
    current_data=current_data
)

drift_test_suite

# Get test results
test_results = drift_test_suite.as_dict()
for test in test_results["tests"]:
    status = "✓ PASS" if test["status"] == "SUCCESS" else "✗ FAIL"
    print(f"{status}: {test['description']}")
```

### 7.6 SHAP for Model Interpretability

[SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/) explains model predictions by computing feature contributions using cooperative game theory.

#### Installation

```bash
pip install shap
```

#### Basic SHAP Explanation

```python
import shap
import matplotlib.pyplot as plt

# Train a model for explanation
X_train_df = pd.DataFrame(X_train, columns=["WTI_Oil"])
model_exp = LinearRegression()
model_exp.fit(X_train_df, y_train)

# Create SHAP explainer
explainer = shap.Explainer(model_exp, X_train_df)
shap_values = explainer(X_train_df)

# Summary plot: feature importance
shap.summary_plot(shap_values, X_train_df, show=False)
plt.title("SHAP Feature Importance Summary")
plt.tight_layout()
plt.show()
```

#### Waterfall Plot (Single Prediction)

```python
# Explanation for a single prediction (e.g., oil price = $85)
single_example = X_train_df.iloc[[0]]
shap_values_single = explainer(single_example)

shap.waterfall_plot(
    shap_values_single[0],
    max_display=10,
    show=False
)
plt.title(f"SHAP Waterfall: Oil=${single_example['WTI_Oil'].values[0]:.1f}")
plt.tight_layout()
plt.show()
```

#### SHAP Dependence Plot

```python
# How does the feature value affect the prediction?
shap.dependence_plot(
    "WTI_Oil", shap_values.values, 
    X_train_df, show=False
)
plt.title("SHAP Dependence: Feature Value vs. Impact")
plt.tight_layout()
plt.show()
```

#### SHAP + Evidently: Interpreting Drift

When drift is detected, SHAP helps answer *which features are driving the drift*:

```python
def interpret_drift_with_shap(reference_data, current_data, model, feature_names):
    """
    Use SHAP to explain drift: compare SHAP values between reference and current.
    Large changes in SHAP values indicate features whose importance has shifted.
    """
    # SHAP values for reference
    ref_explainer = shap.Explainer(model, reference_data)
    ref_shap = ref_explainer(reference_data)
    
    # SHAP values for current
    curr_shap = ref_explainer(current_data)
    
    # Compare mean absolute SHAP values
    ref_importance = np.abs(ref_shap.values).mean(axis=0)
    curr_importance = np.abs(curr_shap.values).mean(axis=0)
    
    importance_change = pd.DataFrame({
        "Feature": feature_names,
        "Reference SHAP (|impact|)": ref_importance,
        "Current SHAP (|impact|)": curr_importance,
        "Change (%)": ((curr_importance - ref_importance) / (ref_importance + 1e-10) * 100)
    }).sort_values("Change (%)", ascending=False)
    
    return importance_change

feature_importance_change = interpret_drift_with_shap(
    X_train_df,
    pd.DataFrame(X_test, columns=["WTI_Oil"]),
    final_model,
    ["WTI_Oil"]
)
print(feature_importance_change)
```

### 7.7 Putting It All Together: Unified Drift Monitoring Dashboard

```python
class UnifiedDriftMonitor:
    """
    Comprehensive monitoring pipeline supporting both linear and cluster models.
    Combines: statistical tests, Evidently reports, SHAP explanations,
    model-specific drift detectors (rolling regression for linear,
    centroid/assignment tracking for cluster models).
    """
    
    def __init__(self, model, model_type="linear", reference_data=None, 
                 feature_names=None, cluster_model=None):
        self.model = model
        self.model_type = model_type
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.cluster_model = cluster_model
        self.drift_history = []
    
    def run_full_monitoring(self, current_data, y_current=None):
        """Run all applicable drift checks based on model type."""
        results = {"timestamp": pd.Timestamp.now()}
        
        # 1. Universal: Evidently Data Drift
        if self.reference_data is not None:
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(
                reference_data=self.reference_data,
                current_data=pd.DataFrame(current_data, columns=self.feature_names),
                column_mapping={"numerical_features": self.feature_names}
            )
            drift_metrics = drift_report.as_dict()
            results["drift_share"] = drift_metrics["metrics"][0]["result"]["drift_share"]
            results["drift_alert"] = results["drift_share"] > 0.2
        
        # 2. Linear model specific
        if self.model_type == "linear" and y_current is not None:
            y_pred = self.model.predict(current_data)
            residuals = y_current.flatten() - y_pred.flatten()
            
            # Residual mean shift
            _, t_p = stats.ttest_1samp(residuals, 0)
            results["residual_bias_p"] = t_p
            results["residual_bias_alert"] = t_p < 0.05
            
            # CUSUM
            cusum = np.cumsum((residuals - np.mean(residuals)) / np.std(residuals))
            results["cusum_alert"] = np.max(np.abs(cusum)) > 5
        
        # 3. Cluster model specific
        if self.model_type == "cluster" and self.cluster_model is not None:
            curr_labels = self.cluster_model.predict(current_data)
            ref_labels = self.cluster_model.predict(self.reference_data)
            
            # Assignment distribution drift
            ref_dist = pd.Series(ref_labels).value_counts(normalize=True)
            curr_dist = pd.Series(curr_labels).value_counts(normalize=True)
            assignment_drift = sum(abs(curr_dist.get(c, 0) - ref_dist.get(c, 0)) 
                                  for c in set(ref_dist.index) | set(curr_dist.index))
            results["cluster_assignment_drift"] = assignment_drift
            results["cluster_drift_alert"] = assignment_drift > 0.3
        
        # 4. SHAP importance stability (any model)
        if self.reference_data is not None and self.feature_names is not None:
            try:
                explainer = shap.Explainer(self.model, self.reference_data)
                ref_shap = explainer(self.reference_data)
                curr_shap = explainer(pd.DataFrame(current_data, columns=self.feature_names))
                shap_stability = np.corrcoef(
                    np.abs(ref_shap.values).mean(axis=0),
                    np.abs(curr_shap.values).mean(axis=0)
                )[0, 1]
                results["shap_stability"] = shap_stability
                results["shap_alert"] = shap_stability < 0.8
            except Exception:
                pass  # SHAP may fail on some models
        
        self.drift_history.append(results)
        return results

# Example: Linear model monitoring
linear_monitor = UnifiedDriftMonitor(
    model=final_model,
    model_type="linear",
    reference_data=X_train_df,
    feature_names=["WTI_Oil"]
)
linear_result = linear_monitor.run_full_monitoring(
    X_test, y_test.reshape(-1, 1)
)
print("Linear Model Drift Assessment:")
print(json.dumps({k: v for k, v in linear_result.items() 
                  if not isinstance(v, (pd.Timestamp,))}, indent=2, default=str))

# Example: Cluster model monitoring
cluster_monitor = UnifiedDriftMonitor(
    model=cluster_model,
    model_type="cluster",
    reference_data=X_train,
    feature_names=["WTI_Oil"],
    cluster_model=cluster_model
)
cluster_result = cluster_monitor.run_full_monitoring(X_test)
print("\nCluster Model Drift Assessment:")
print(json.dumps({k: v for k, v in cluster_result.items()
                  if not isinstance(v, (pd.Timestamp,))}, indent=2, default=str))
```

### 7.8 Drift Detection Summary: Linear vs. Cluster Models

| Drift Type | Linear Models | Cluster Models |
|---|---|---|
| **Feature drift** | PSI, KS-test, Evidently | PSI, KS-test, Evidently ✓ same |
| **Prediction drift** | Residual mean/variance monitoring | N/A (unsupervised) |
| **Parameter drift** | Rolling coefficients, Chow test | Centroid drift tracking |
| **Structural drift** | CUSUM, Durbin-Watson | Silhouette degradation monitoring |
| **Distribution shift** | Leverage (Mahalanobis distance) | Cluster assignment distribution |
| **Concept drift** | Coefficient stability (β changes) | Per-cluster feature drift |
| **Explainability** | SHAP coefficient decomposition | SHAP per-cluster centroid importance |
| **Tools** | statsmodels, CUSUM, Evidently | KMeans shift analysis, Evidently |

---

## Monitoring — A Beginner's Guide

Monitoring is essential once a model serves real users. Start simple and add complexity as you learn:

- **What to monitor:** input feature distributions, prediction distributions, latency, error metrics (MAE/RMSE), and business KPIs.
- **Data drift vs concept drift:** data drift means inputs changed; concept drift means the relationship input→target changed. Both require investigation but different responses.
- **When to alert:** combine statistical tests (PSI, KS), effect-size thresholds, and rolling performance metrics. Avoid single noisy alerts — look for sustained signals.
- **Tools:** Evidently for automated reports, Prometheus/Grafana for metrics, SHAP for interpretability.

## Overfitting — Quick Explanation and Mitigations

**Overfitting** happens when a model captures noise in the training data rather than the true signal. It leads to strong training performance but poor generalisation.

**Signs:** large train/validation gap, unstable validation across folds, very complex models for small datasets.

**Fixes:**
- Get more data or augment existing data.
- Use regularisation (L1/L2, ElasticNet), simpler models, or reduce features.
- Use proper validation (time-series splits for temporal data) and learning curves.
- Prevent data leakage (no future-derived features in training).

## Clusterisation (Clustering) — Practical Notes

- **Purpose:** group similar observations; clustering is unsupervised and does not provide labels.
- **Preprocessing:** scale features, remove outliers, choose informative features.
- **Choosing k:** use silhouette score, Davies–Bouldin, Calinski–Harabasz, inertia (elbow) and gap statistic — prefer consensus across metrics.
- **Validate:** bootstrap stability, visualise with PCA/t-SNE, and profile cluster statistics.

## Linear Drift & Statistical Tests — Intuition

- **PSI (Population Stability Index):** compares binned distributions; interpretable thresholds (PSI &lt; 0.1 OK, 0.1–0.25 caution, &gt;0.25 likely drift).
- **KS test:** non-parametric distribution comparison with p-values for significance.
- **KL divergence:** measures divergence between distributions; useful for ranking but unbounded.
- **Residual tests for supervised models:** t-tests for mean shifts, F-tests for variance changes, Durbin–Watson for autocorrelation, and CUSUM/Chow tests for structural breaks.

Use these together rather than in isolation. Statistical significance (p-values) without effect sizes can be misleading for very large sample sizes.

Note: The notebook now includes a dedicated "Overfitting for Time Series" session with expanding-window learning-curve visuals and additional supervised/unsupervised statistical test examples for beginners.


## Summary

The end-to-end ML pipeline in the mlops-engineering project demonstrates:

| Stage | Tool / Technique | Key Takeaway |
|---|---|---|
| **Data Engineering** | yfinance, pandas | Clean, validate, split chronologically |
| **Feature Engineering** | lags, rolling windows, returns | Transform raw data into predictive signals |
| **Model Selection** | Linear Regression, comparison | Simpler is often better for production |
| **Evaluation (Linear)** | Coefficient analysis, VIF, heteroscedasticity tests, DW, Cook's distance | Verify statistical assumptions |
| **Evaluation (Cluster)** | Silhouette, DB Index, CH Index, gap statistic, stability analysis | Internal metrics + stability determine quality |
| **Optimisation** | Regularisation, hyperparameter tuning | Balance accuracy vs. interpretability |
| **Drift (Linear)** | Rolling regression, CUSUM, Chow test, residual monitoring, leverage | Track parameter stability + assumption violations |
| **Drift (Cluster)** | Centroid tracking, assignment drift, silhouette degradation, per-cluster feature PSI | Track structural coherence |
| **Drift (Universal)** | Evidently AI, PSI, KS-test | Monitor continuously; trigger retraining |
| **Interpretability** | SHAP | Understand *why* predictions change |

### Next Steps

- Add Evidently drift monitoring to the FastAPI service
- Integrate MLflow for experiment tracking and model registry
- Add automated retraining pipeline triggered by drift alerts
- Store production data in MinIO for model retraining

---

## References

- [mlops-engineering GitHub Repository](https://github.com/yev-dev/mlops-engineering)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)