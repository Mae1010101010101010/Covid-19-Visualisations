#“Can we predict a country's COVID-19 death rate per million using its case rate, vaccination progress, and testing strategy?”

import pandas as pd

file_path = r"C:\Users\mouna\OneDrive\Desktop\CovidData.csv"

df = pd.read_csv(file_path)

print(df.head())

import matplotlib.pyplot as plt


features = [
    'total_cases_per_million',
    'people_fully_vaccinated_per_hundred',
    'positive_rate',
    'total_tests_per_thousand',
    'population_density',
    'median_age',
    'gdp_per_capita',
]

df_model = df[features + ['total_deaths_per_million']].dropna()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Prepare your data
X = df_model[features]
y = df_model['total_deaths_per_million']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction (this is the missing line!)
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R2 Score:", r2)
print("RMSE:", rmse)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Deaths per Million")
plt.ylabel("Predicted Deaths per Million")
plt.title("Model Predictions vs Actual")
plt.grid(True)
plt.show()


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

importances = rf.feature_importances_
plt.barh(features, importances)
plt.title("Feature Importances")
plt.show()
