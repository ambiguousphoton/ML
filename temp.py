import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.impute import SimpleImputer as Imputer
dataset = pd.read_csv("C:/Users/microsoft/Desktop/#/python programs/ML/1/Data.csv")

x = dataset.iloc[:, :-1].value

y = dataset.iloc[:, 3].value
narayan ="narayan narayan narayan narayan"
print(narayan)


imputer = Imputer.fit(x[:,1:])
imputer = imputer.fit(x[:, 1:3])
