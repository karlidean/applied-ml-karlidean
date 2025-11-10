"""ml01.py - Just the code.

This script is a simple example of a linear regression model using the California Housing dataset.
Some charting is commented out after evaluation.
The dataset is loaded, explored, and visualized.
Features are selected and a linear regression model is trained.
The model is evaluated using R², MAE, and RMSE.
"""

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

#########################################
# Section 1. Load and explore the dataset.
#########################################

# Load the California Housing dataset
housing_data = fetch_california_housing(as_frame=True, return_X_y=True)

# Separate features and target
X_df: pd.DataFrame = cast("pd.DataFrame", housing_data[0])
y_ser: pd.Series = cast("pd.Series", housing_data[1])

# Type hints for clarity and tooling
X: pd.DataFrame = X_df
y: pd.Series = y_ser

# Combine for exploration
data_frame: pd.DataFrame = X.copy()
data_frame["MedHouseVal"] = y

# Display the first 10 rows of the dataset
data_frame.head(10)

# Check data types and missing values
data_frame.info()

# Summary statistics
data_frame.describe()

# Check for missing values
data_frame.isnull().sum()

#########################################
# Section 2. Visualize Feature Distributions
#########################################

# Histograms for all numeric columns
data_frame.hist(bins=30, figsize=(12, 8))
plt.show()

# Boxenplots for all numeric columns
# for column in data_frame.columns:
#     plt.figure(figsize=(6, 4))
#     sns.boxenplot(data=data_frame[column])
#     plt.title(f'Boxenplot for {column}')
#     plt.show()

# Pairplot for all numeric columns
# With 9 columns, there will be 9x9 = 81 plots, so this is commented out after analysis.
# sns.pairplot(data_frame)
# plt.show()

#########################################
# Section 3. Feature Selection and Justification
#########################################

features: list = ["MedInc", "AveRooms"]
target: str = "MedHouseVal"
df_X = data_frame[features]  #  # noqa: N816
df_y = data_frame[target]

#########################################
# Section 4. Train a Linear Regression Model
#########################################

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

# make an instance of the model, fit the data, then predict test y values
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model using R², MAE, and RMSE.

r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.2f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")

"""
SAMPLE OUTPUT:

RangeIndex: 20640 entries, 0 to 20639
Data columns (total 9 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -------
 0   MedInc       20640 non-null  float64
 1   HouseAge     20640 non-null  float64
 2   AveRooms     20640 non-null  float64
 3   AveBedrms    20640 non-null  float64
 4   Population   20640 non-null  float64
 5   AveOccup     20640 non-null  float64
 6   Latitude     20640 non-null  float64
 7   Longitude    20640 non-null  float64
 8   MedHouseVal  20640 non-null  float64
dtypes: float64(9)
memory usage: 1.4 MB
R²: 0.46
MAE: 0.62
RMSE: 0.84

"""
