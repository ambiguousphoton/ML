# Random Forest Regression
### non continues non linear model 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 3].values

# Fitting the decision tree regressor to dataset 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(x, y)

y_prd = regressor.predict([[6.5]])

# Visualisation
x_grid = np.arange(min(x),max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title("Random Forest Regression")
plt.show()
