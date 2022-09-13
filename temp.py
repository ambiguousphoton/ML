Name = "Template for ML by Vyoam"
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
dataset = pd.read_csv("Data.csv")

x = dataset.iloc[:, :-1].value

y = dataset.iloc[:, 3].value
narayan ="narayan narayan narayan narayan"
print(narayan)


imputer = SimpleImputer(missing_values=np.nan, strategy='mean' , fill_value=None, verbose="deprecated", copy=True, add_indicator=False)
imputer = imputer.fit(x[:, 1:]) 
x[:, 1:3] =  imputer.transform(x[:, 1:3])  
labelEncoder_x = LabelEncoder()
x[:, 0] = labelEncoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categories = [0])
x =onehotencoder.fit_transform(x[0,:]).toarray()





