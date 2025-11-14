## Imports
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


## Section 1 - Import and Inspect the Data
# Load Titanic dataset from seaborn and verify
titanic = sns.load_dataset("titanic")
titanic.head()


## Section 2 - Data Exploration and Preparation
```shell
titanic['age'].fillna(titanic['age'].median(), inplace=True) # Imputes missing values for the median

titanic = titanic.dropna(subset=['fare']) # Drops rows that don't have a fare (fare is missing) 
  # I could impute this if I wanted to, but I am not choosing that route.

titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1 # Creating numeric variables
  # Consider doing this with sex as well, if you believe it will help your model.
```


## Section 3 - Feature Selection and Justification
```shell
# Case 1. age
X1 = titanic[['age']]
y1 = titanic['fare']

# Case 2. family_size
X2 = titanic[['family_size']]
y2 = titanic['fare']

# Case 3. age, family_size
X3 = titanic[['age', 'family_size']]
y3 = titanic['fare']

# Case 4. ??? (THIS IS MY CHOICE)
# TODO: Select my feature/features (If I choose sex, I have to change the categorical data into numeric values in section 2!)
X4 = titanic[[???]]
y4 = titanic['fare']
```


## Reflection Questions (markdown)
1. Why might these features affect a passenger’s fare: **Riders that are solo and wealthy are more inclined to pay a higher fare for a higher class, while family sizes that are higher are more inclined to purchase lower fares and lower classes to get their whole family on board.**
2. List all available features:
3. Which other features could improve predictions and why:
4. How many variables are in your Case 4:
5. Which variable(s) did you choose for Case 4 and why do you feel those could make good inputs: 


## Section 4 - Training a Regression Model (Linear Regression)
### 4.1 - Splitting the Data
```shell
# Creating Test Group for Case 1
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=123)

# Creating Test Group for Case 2
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=123)

# Creating Test Group for Case 3
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=123)

# Creating test Group for Case 4
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=123)
```


### 4.2 - Training and Evaluating Linear Regression Models
```shell
lr_model1 = LinearRegression().fit(X1_train, y1_train)
lr_model2 = LinearRegression().fit(X2_train, y2_train)
lr_model3 = LinearRegression().fit(X3_train, y3_train)
lr_model4 = LinearRegression().fit(X4_train, y4_train)
```

# Predictions

```shell
# Case 1
y_pred_train1 = lr_model1.predict(X1_train)
y_pred_test1 = lr_model1.predict(X1_test)

# Case 2
y_pred_train2 = lr_model2.predict(X2_train)
y_pred_test2 = lr_model2.predict(X2_test)

# Case 3
y_pred_train3 = lr_model3.predict(X3_train)
y_pred_test3 = lr_model3.predict(X3_test)

# Case 4
y_pred_train4 = lr_model4.predict(X4_train)
y_pred_test4 = lr_model4.predict(X4_test)
```

### 4.3 - Reporting Performance
```shell
# Case 1
print("Case 1: Training R²:", r2_score(y1_train, y1_pred_train))
print("Case 1: Test R²:", r2_score(y1_test, y1_pred_test))
print("Case 1: Test RMSE:", mean_squared_error(y1_test, y1_pred_test, squared=False))
print("Case 1: Test MAE:", mean_absolute_error(y1_test, y1_pred_test))

# Case 2
print("Case 2: Training R²:", r2_score(y2_train, y2_pred_train))
print("Case 2: Test R²:", r2_score(y2_test, y2_pred_test))
print("Case 2: Test RMSE:", mean_squared_error(y2_test, y2_pred_test, squared=False))
print("Case 2: Test MAE:", mean_absolute_error(y2_test, y2_pred_test))

# Case 3
print("Case 1: Training R²:", r2_score(y3_train, y3_pred_train))
print("Case 1: Test R²:", r2_score(y3_test, y3_pred_test))
print("Case 1: Test RMSE:", mean_squared_error(y3_test, y3_pred_test, squared=False))
print("Case 1: Test MAE:", mean_absolute_error(y3_test, y3_pred_test))

# Case 4
print("Case 1: Training R²:", r2_score(y4_train, y4_pred_train))
print("Case 1: Test R²:", r2_score(y4_test, y4_pred_test))
print("Case 1: Test RMSE:", mean_squared_error(y4_test, y4_pred_test, squared=False))
print("Case 1: Test MAE:", mean_absolute_error(y4_test, y4_pred_test))
```

## Reflection Questions (markdown)
#Compare the train vs test results for each.
1. Did Case 1 overfit or underfit? Explain:
2. Did Case 2 overfit or underfit? Explain:
3. Did Case 3 overfit or underfit? Explain:
4. Did Case 4 overfit or underfit? Explain:

# Adding Age
1. Did adding age improve the model:
2. Propose a possible explanation (consider how age might affect ticket price, and whether the data supports that): 

# Worst
1. Which case performed the worst:
2. How do you know: 
3. Do you think adding more training data would improve it (and why/why not): 

# Best
1. Which case performed the best:
2. How do you know: 
3. Do you think adding more training data would improve it (and why/why not): 


## Section 5 - Comparing Alternative Models (Choose the Best Case to Continue and Edit Following Code if it is NOT Case 1)
### TODO: Evaluate your models and choose the best one!

### 5.1 - Ridge Regression (L2 Penalty)
```shell
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X1_train, y1_train)
y_pred_ridge = ridge_model.predict(X1_test)
```
### 5.2 - Elastic Net (L1 + L2 Combined)
```shell
elastic_model = ElasticNet(alpha=0.3, l1_ratio=0.5)
elastic_model.fit(X1_train, y1_train)
y_pred_elastic = elastic_model.predict(X1_test)
```
### 5.3 - Polynomial Regression
# Set up the poly inputs
```shell
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X1_train)
X_test_poly = poly.transform(X1_test)
```
# Use the poly inputs in the LR model
```shell
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y1_train)
y_pred_poly = poly_model.predict(X1_test_poly)
```
### 5.4 - Visualize Polynomial Cubic Fit (for 1 Input)
#### Choose one of the cases with 1 input feature and plot it with this!
```shell
plt.scatter(X1_test, y1_test, color='blue', label='Actual')
plt.scatter(X1_test, y_pred_poly, color='red', label='Predicted (Poly)')
plt.legend()
plt.title("Polynomial Regression: Age vs Fare")
plt.show()
```

## Reflection Questions (markdown)
1. What patterns does the cubic model seem to capture:
2. Where does it perform well or poorly:
3. Did the polynomial fit outperform linear regression:
4. Where (on the graph or among which kinds of data points) does it fit best:


### 5.5 - Compare All Models
```shell
def report(name, y_true, y_pred):
    print(f"{name} R²: {r2_score(y_true, y_pred):.3f}")
    print(f"{name} RMSE: {mean_squared_error(y_true, y_pred, squared=False):.2f}")
    print(f"{name} MAE: {mean_absolute_error(y_true, y_pred):.2f}\n")

report("Linear", y1_test, y1_pred_test)
report("Ridge", y1_test, y_pred_ridge)
report("ElasticNet", y1_test, y_pred_elastic)
report("Polynomial", y1_test, y_pred_poly)
```

### 5.6 - Visualize Higher Order Polynomials (For Same 1 Input Case)
# Set up the poly inputs
```shell
poly = PolynomialFeatures(degree=3) # TODO: Change this degree!
X_train_poly = poly.fit_transform(X1_train)
X_test_poly = poly.transform(X1_test)

# Use the poly inputs in the LR model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y1_train)
y_pred_poly = poly_model.predict(X1_test_poly)

# Visualize It
plt.scatter(X1_test, y1_test, color='blue', label='Actual')
plt.scatter(X1_test, y_pred_poly, color='red', label='Predicted (Poly)')
plt.legend()
plt.title("Polynomial Regression: Age vs Fare")
plt.show()
```
## Reflection Questions (markdown)
1. Which Option worked better for the polynomial model? The initial cubic one or the new (higher degree) one?

## Section 6 - Final Thoughts and Insights
### 6.1 - Summarize Findings
1. What features were most useful?
2. What regression model performed best?
3. How did model complexity or regularization affect results?

### 6.2 - Discuss Challenges
1. Was fare hard to predict? Why or why not?
2. Did skew or outliers impact the models?

### 6.3 - Optional Next Steps
1. Try different features besides the ones used in this project.
2. Try predicting age instead of fare.
3. Explore log transformation of fare to reduce skew.
