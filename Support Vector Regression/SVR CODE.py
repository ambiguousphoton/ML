import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,2:3].values
y = dataset.iloc[:,2:4].values

# Feature Scaling cause it is not preavailable in SVR
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
y = y[:,1]

# Fitting regressor to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)
# arr = np.array([[6.5],[0]])
# # arr.reshape(-1,1)
# arr2 = regressor.predict(sc_x.transform(arr))
# arr2 = [arr2]
# y_prd = sc_y.inverse_transform(np.array(arr2))

plt.scatter(x, y, color ='blue')
plt.plot(x, regressor.predict(x), color ='red')
plt.title('Truth or Bluff')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

