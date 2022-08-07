import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,2:3].values
y = dataset.iloc[:,3].values

# Linear Regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

# Fitting Polynomial Linear Regressor to the dataset
from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(degree=4)
x_poly = pol.fit_transform(x)
linreg = LinearRegression()
linreg.fit(x_poly,y)

# visual linear regression 
plt.scatter(x, y, color = 'red')
plt.title('Truth or Bluff')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.plot(x, lr.predict(x), color = 'blue')
plt.show()

# visualising polinomial linear regression
plt.scatter(x, y, color ='blue')
plt.plot(x, linreg.predict(pol.fit_transform(x)), color ='red')
plt.title('Truth or Bluff')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Predicting  new  result  with  linear model
arr = [[6.5], [0]]
lr.predict(arr)
