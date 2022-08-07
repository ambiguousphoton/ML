# Multiple linear regression
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Import dataset
dataset = pd.read_csv("startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Creating the dummy variables 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# encoding
from sklearn.compose import ColumnTransformer 
ct  =  ColumnTransformer([("om",OneHotEncoder(),[3])],remainder="passthrough")
ct.fit_transform(x)
x[:,3] = LabelEncoder().fit_transform(x[:,3])

# Splitting the dataset into training set and test dataset 
from sklearn.model_selection import train_test_split 
X_train , X_test , y_train , y_test = train_test_split(x, y ,test_size=0.20) 

# Fitting multiple linear regression to training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Pridiction of test sets 
y_predict = regressor.predict(X_test)

# building the optimal model using backward elimination
import statsmodels.api as sm
#---->try 1 
x = np.append(arr = np.ones((50,1)).astype(int), values=x ,  axis=1)
X_opt = x[:,[0,1,2,3,4]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y,exog=X_opt)
regressor_OLS.fit()
result = regressor_OLS.fit().summary()

#----->try 2
X_opt = x[:,[0,1,3,4]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y,exog=X_opt)
regressor_OLS.fit()
result = regressor_OLS.fit().summary()

#------>try 3
X_opt = x[:,[0,1,3]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y,exog=X_opt)
regressor_OLS.fit()
result = regressor_OLS.fit().summary()

#------>try 3
X_opt = x[:,[0,1]]
X_opt = np.array(X_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y,exog=X_opt)
regressor_OLS.fit()
result = regressor_OLS.fit().summary()
print(result)



