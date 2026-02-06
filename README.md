# Flight Price Prediction using Regression Models

## Project Overview
This project aims to predict **flight ticket prices** using multiple supervised **regression models**.  
It demonstrates a complete **machine learning pipeline**, including data preprocessing, feature encoding, model training, and evaluation using standard regression metrics.


---

## Data Preprocessing

### 1. Importing Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
```

---

### 2. Checking Missing Values
Missing values are identified to ensure data quality.

```python
df.isnull().sum()
```

**Why?**  
Missing values can bias model training or cause runtime errors. They must be handled before encoding or modeling.

---

### 3. Ordinal Encoding
Used for **ordered categorical features** such as number of stops.

Example:
- Non-stop < 1 Stop < 2 Stops < 3+ Stops

```python
ordinal_cols = ["Total_Stops"]
encoder = OrdinalEncoder()
df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])
```

**Why?**  
Ordinal encoding preserves the natural order of categories, which is important for regression models.

---

### 4. One-Hot Encoding
Applied to **nominal categorical features** such as airline, source, and destination.

```python
nominal_cols = ["Airline", "Source", "Destination"]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
```

**Why?**  
Nominal features do not have an order, so one-hot encoding prevents the model from assuming false relationships.

---

### 5. Train-Test Split
```python
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why?**  
Separates unseen data to evaluate generalization performance.

---

## Machine Learning Models Used

### 1. Linear Regression
*Baseline model that assumes a linear relationship between features and price.*

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
```

---

### 2. Ridge Regression (L2 Regularization)
*Adds L2 regularization to reduce overfitting caused by large coefficients.*

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
```

---

### 3. Lasso Regression (L1 Regularization)
*Uses L1 regularization and can eliminate less important features.*

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
```

---

### 4. ElasticNet Regression
*Combines Ridge and Lasso regularization for balanced feature selection.*

```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)
```

---

### 5. Decision Tree Regressor
*Learns non-linear decision rules by recursively splitting the data.*

```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

---

### 6. Random Forest Regressor
*Ensemble of multiple decision trees that improves accuracy and stability.*

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

---

### 7. AdaBoost Regressor
*Boosting-based model that focuses on correcting previous prediction errors.*

```python
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
```

---

## Model Evaluation Metrics

### R² Score
Measures how well the model explains variance in flight prices.

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
```

---

### Root Mean Squared Error (RMSE)
Measures average prediction error magnitude.

```python
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred, squared=False)
```

---

## Interpretation
- Higher **R²** → Better explanatory power
- Lower **RMSE** → More accurate predictions
- Ensemble models typically outperform linear models due to non-linear pattern capture

---

## Requirements
```bash
pip install pandas numpy scikit-learn
```

---

## Author
**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  

