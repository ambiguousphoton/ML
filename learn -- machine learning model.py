import numpy as np 
import pandas as pd  #library handling datasets
from sklearn.impute import SimpleImputer #
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

#including dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
narayan ="narayan narayan narayan narayan"
print(narayan)

#handling missing data by replacing it with mean values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean' , fill_value=None, verbose="deprecated", copy=True, add_indicator=False)# imputer objecet
imputer = imputer.fit(x[:, 1:]) 
x[:, 1:3] = imputer.transform(x[:, 1:3])   # fitting new 

#Encoding + catagorising data
from sklearn.compose import ColumnTransformer
ct =ColumnTransformer([("",OneHotEncoder(),[0])],remainder="passthrough")
ct.fit_transform(x)

#---> encoding
labelEncoder_x = LabelEncoder()
x[:, 0] = labelEncoder_x.fit_transform(x[:,0]) # ----> fit

#---> catagorising data
ct = ColumnTransformer([("country",OneHotEncoder(),[0])], remainder='passthrough')
x = ct.fit_transform(x)

#Spliting the data-set into the training set and test set
from sklearn.model_selection import train_test_split 
X_train , X_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state = 0 )

# ---> 0.25 is better test size 