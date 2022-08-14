import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("dataset.csv")
x = dataset.iloc[:,[29]].values 
y = dataset.iloc[:,[46]].values

# teq vs sun radius srad

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

# visual linear regression 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.title('Linear Prediction (train)')
plt.ylabel("49")
plt.xlabel("48")
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.title('Linear Prediction (test)')
plt.ylabel("49")
plt.xlabel("48")
plt.show()
