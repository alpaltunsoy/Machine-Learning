# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:56:43 2023

@author: Alp Altunsoy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("customers.csv")
X= dataset.iloc[:,3:].values

#kmeans

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = "k-means++")
kmeans.fit(X)
print(kmeans.cluster_centers_)

sonuclar =[]

#finding best cluster number WCSS
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init = "k-means++",random_state = 123)
    kmeans.fit(X)
    #Sum of squared distances of samples to their closest cluster center WSCC
    sonuclar.append(kmeans.inertia_)
    
    
    
plt.plot(range(1,10),sonuclar)
plt.show()

#after finding best cluster number
kmeans = KMeans(n_clusters = 4, init = "k-means++")
y_predict_k= kmeans.fit_predict(X)

#scatting
plt.scatter(X[y_predict_k ==0,0], X[y_predict_k == 0,1], s=100, c="red")
plt.scatter(X[y_predict_k ==1,0], X[y_predict_k == 1,1], s=100, c="blue")
plt.scatter(X[y_predict_k ==2,0], X[y_predict_k == 2,1], s=100, c="green")
plt.scatter(X[y_predict_k ==3,0], X[y_predict_k == 3,1], s=100, c="black")
plt.show()



#Aggloramoticvr Clustering
from sklearn.cluster import AgglomerativeClustering

ac= AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage ="ward")
y_predict = ac.fit_predict(X)
print(y_predict)

#scatting
plt.scatter(X[y_predict ==0,0], X[y_predict == 0,1], s=100, c="red")
plt.scatter(X[y_predict ==1,0], X[y_predict == 1,1], s=100, c="blue")
plt.scatter(X[y_predict ==2,0], X[y_predict == 2,1], s=100, c="green")
plt.show()

#dendogram
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.show()

    