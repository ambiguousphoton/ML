import numpy as np
import matplotlib.pyplot as plt 
import  pandas as pd
By = "Vyoam"
dataset = pd.read_csv("Data.csv")

matrix = dataset.iloc[:, :-1].values

arr = dataset.iloc[:,3].values
