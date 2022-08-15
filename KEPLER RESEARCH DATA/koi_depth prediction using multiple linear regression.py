# MULTIPLE LINEAR REGRESSION 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("dataset.csv")
x = dataset.iloc[:,[26,19,44]].values 
y = dataset.iloc[:,[22]].values

# planet radius (prad) vs sun radius (srad)

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean' , fill_value=None, verbose="deprecated", copy=True, add_indicator=False)# imputer objecet
imputer = imputer.fit(y[:, :]) 
y[:, :] = imputer.transform(y[:, :]) 

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean' , fill_value=None, verbose="deprecated", copy=True, add_indicator=False)# imputer objecet
imputer = imputer.fit(x[:, :]) 
x[:, :] = imputer.transform(x[:, :]) 

# #Encoding + catagorising data
# from sklearn.preprocessing import LabelEncoder , OneHotEncoder
# from sklearn.compose import ColumnTransformer
# ct =ColumnTransformer([("",OneHotEncoder(),[0])],remainder="passthrough")
# ct.fit_transform(x)

# #---> encoding
# labelEncoder_x = LabelEncoder()
# x[:, 0] = labelEncoder_x.fit_transform(x[:,0]) # ----> fit

# #---> catagorising data
# ct = ColumnTransformer([("country",OneHotEncoder(),[0])], remainder='passthrough')
# x = ct.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x,y,train_size=0.25) 

# Linear Regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

predicted_train = lr.predict(X_train)
predicted_test = lr.predict(X_test)
# visual linear regression 
x_grid = np.arange(min(x),max(x), 0.001)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(x_grid, lr.predict(x_grid), color='blue')
plt.title('Multiple Linear Prediction (train)')
plt.ylabel("29")
plt.xlabel("46")
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(x_grid, lr.predict(x_grid), color='blue')
plt.title('Multiple Linear Prediction (test)')
plt.ylabel("29")
plt.xlabel("46")
plt.show()
