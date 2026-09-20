# CO₂ Emissions Analysis & Prediction 🚗💨

## Problem Statement

This project analyzes a dataset of vehicle specifications, engine characteristics, fuel types, and emission levels to uncover patterns linking vehicle attributes to CO₂ emissions. Through exploratory data analysis and predictive modeling, the goal is to identify the major contributors to emissions and provide data-driven insights that can support automotive design choices and environmental policy decisions.

## Dataset

The dataset (`CO2_Emissions.csv`) contains **7,385 records** across **12 features**, covering vehicle make, model, class, engine specs, fuel type, fuel consumption (city/highway/combined), and CO₂ emissions.

**Source:** [CO2 Emission by Vehicles — Kaggle](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles/data)

| Feature | Description |
|---|---|
| Make | Vehicle manufacturer |
| Model | Vehicle model |
| Vehicle Class | Category (SUV, Compact, Pickup, etc.) |
| Engine Size (L) | Engine displacement in liters |
| Cylinders | Number of cylinders |
| Transmission | Transmission type/code |
| Fuel Type | Fuel category (X, Z, E, D, N) |
| Fuel Consumption City (L/100 km) | City fuel consumption |
| Fuel Consumption Hwy (L/100 km) | Highway fuel consumption |
| Fuel Consumption Comb (L/100 km) | Combined fuel consumption |
| Fuel Consumption Comb (mpg) | Combined fuel consumption in mpg |
| **CO2 Emissions (g/km)** | **Target variable** |

No missing values were present in the dataset.

## Approach

### 1. Exploratory Data Analysis
- Univariate analysis of every feature (distribution plots, box plots, bar charts) and its relationship with CO₂ emissions.
- Bivariate analysis to understand redundancy between related features (e.g., `Model` vs `Make`, fuel consumption metrics vs each other).
- Correlation matrix and pairplots to assess linear relationships among numeric features.

### 2. Feature Engineering & Selection
- **Dropped `Fuel Consumption Comb (mpg)`** — mathematically derivable from `Fuel Consumption Comb (L/100 km)` (correlation ≈ 0.999 after unit conversion); retaining both would be redundant.
- **Dropped `Fuel Consumption Comb (L/100 km)`** — shown via regression to be a near-exact weighted average of City and Highway consumption (R² > 0.999), confirming redundancy.
- **Dropped `Model`** — 2,053 unique values, most occurring only once or twice; one-hot encoding would add thousands of sparse features and risk overfitting. `Make` was retained as the more generalizable proxy (each model maps to exactly one make).
- **Retained `Engine Size` and `Cylinders` as numeric** — despite being technically categorical/discrete, they preserve a meaningful ordinal relationship without needing re-encoding.

### 3. Multicollinearity Check
- Computed **Variance Inflation Factor (VIF)** across all features.
- Iteratively removed the highest-VIF feature only if the resulting drop in R² stayed under a 2% threshold.
- **`Fuel Consumption City (L/100 km)`** was the only feature removed this way (R² dropped from 0.9943 → 0.9752, within tolerance), since it is highly correlated with `Fuel Consumption Hwy (L/100 km)`.
- Final feature set: `Make`, `Vehicle Class`, `Engine Size(L)`, `Cylinders`, `Transmission`, `Fuel Type`, `Fuel Consumption Hwy (L/100 km)`.

### 4. Modeling
- **Linear Regression** (baseline) using a `ColumnTransformer` pipeline: `StandardScaler` for numeric features, `OneHotEncoder` for categorical features.
- **Polynomial Regression** (degrees 1–5) tested on the shortlisted numeric features to check for non-linear gains.
- Verified core linear regression assumptions: linearity, normality of residuals, homoscedasticity, and independence of errors (autocorrelation ≈ 0.018).

### 5. Feature Importance
- Used **permutation importance** (since one-hot encoded categorical features can't be ranked by raw coefficients) to rank predictors by their true impact on model performance.

## Results

| Model | R² (Train) | R² (Test) |
|---|---|---|
| Linear Regression (all features) | 0.992 | 0.994 |
| Linear Regression (post multicollinearity reduction) | — | 0.975 |
| Polynomial Regression (degree 2) | 0.981 | 0.980 |
| Polynomial Regression (degree 5) | 0.982 | 0.980 |

The degree-1 (plain linear) model was selected as the final model by **Occam's Razor** — higher-degree polynomial features gave negligible improvement at the cost of added complexity.

### Feature Importance (descending)
1. **Fuel Consumption Hwy (L/100 km)** — by far the dominant driver of CO₂ emissions
2. Fuel Type
3. Engine Size (L)
4. Transmission
5. Make
6. Vehicle Class
7. Cylinders

## Key Insights

- Fuel consumption (highway) is the single strongest predictor of CO₂ emissions — far outweighing all other factors combined.
- Engine size and cylinder count correlate with emissions but are largely explained through their relationship with fuel consumption.
- From a policy/design standpoint, **improving fuel efficiency** has by far the greatest leverage on reducing vehicle CO₂ output.

## Tech Stack

- **Python**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Modeling**: scikit-learn (LinearRegression, Pipeline, ColumnTransformer, PolynomialFeatures, permutation_importance)
- **Statistics**: scipy, statsmodels (VIF)

## Repository Structure

```
├── CO2_Emissions.csv          # Raw dataset
├── CO2_Emission_Case_Study.py # Full analysis & modeling script/notebook
└── README.md                  # Project documentation
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy statsmodels
python CO2_Emission_Case_Study.py
```

## Conclusion

A linear regression model achieves an R² of ~0.975–0.99 in predicting CO₂ emissions from vehicle specifications. Fuel consumption on the highway emerged as the dominant factor, followed by fuel type and engine size — insights that can directly inform automotive design priorities and emissions policy.
