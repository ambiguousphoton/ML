# K MEANS CLUSTRING 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

# USING THE ELBOW METHOD TO FIND OPTIMAL NUMBER OF CLUSTERS
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++",max_iter=300,n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("no. of clusters")
plt.ylabel("WCSS")
plt.show() 
# from graph we got optimal no of clusters

#   APPLYING K-MEANS TO DATASET
kmeans = KMeans(n_clusters=5,init="k-means++",max_iter=300,n_init=10,random_state=0)

y_kmeans = kmeans.fit_predict(x)

# VISUALISING THE CLUSTERS
plt.scatter(x[y_kmeans == 0, 0],x[y_kmeans == 0 ,1] ,s = 10, c='red', label='carefull' )
plt.scatter(x[y_kmeans == 1, 0],x[y_kmeans == 1 ,1] ,s = 10, c='blue', label='standard' )
plt.scatter(x[y_kmeans == 2, 0],x[y_kmeans == 2 ,1] ,s = 10, c='green', label='Target' )
plt.scatter(x[y_kmeans == 3, 0],x[y_kmeans == 3 ,1] ,s = 10, c='cyan', label='careless' )
plt.scatter(x[y_kmeans == 4, 0],x[y_kmeans == 4 ,1] ,s = 10, c='magenta', label='Sensible' )
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 30, c = "yellow", label="Centroids")
plt.title("clusters of clients")
plt.xlabel("anual income (ks)")
plt.ylabel("spending score")
plt.legend()
plt.show()