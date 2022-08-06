# Multiple linear regression
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Import dataset
dataset = pd.read_csv("startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Splitting the dataset into training set and test dataset 
from sklearn.model_selection import train_test_split 
X_train , X_test , y_train , y_test = train_test_split(x, y ,test_size=0.25) 


# Creating the dummy variables 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
x[:,3] = LabelEncoder().fit_transform(x[:,3])
x = OneHotEncoder(categories=[3]).fit_transform(x).toarray() 


# Feature Scaling 

