test = "narayan narayan narayan narayan"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('P12-SalaryData.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:, 1]

from sklearn.model_selection import train_test_split
X_train , X_test ,  y_train , y_test = train_test_split(x,y, test_size=1/3,random_state=0)

#---------> making simple regression object and fitting it to dataset 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#---------> pridicted values
y_pridicted = regressor.predict(X_test)

#Visualising the training results
plt.scatter(X_train, y_train , color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel("year of experience")
plt.ylabel("salary")
plt.show()

plt.scatter(X_test,y_test,color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.xlabel("experience")
plt.ylabel("salary")
plt.title("salary vs experience (test set)")
plt.show()

print(test)
print(len(test))
