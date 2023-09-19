# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:26:25 2023

@author: Alp Altunsoy
"""

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

#reading datasets
dataset = pd.read_excel("Iris.xls")

#preprocessing for datas
#slicing
x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4:5].values

#slicing datasets to train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33,random_state=0)

#Scaling datas
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test  = scaler.transform(x_test) 

#classification
#1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
y_pred1 = logistic_reg.predict(X_test)

cm = confusion_matrix(y_test, y_pred1)
print("Logistic Regression\n")
print("Confusion Matrix")
print(cm,end="\n\n")

#2. KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3,metric="minkowski")
knn.fit(X_train,y_train)
y_pred2=knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred2)
print("KNN Classifier\n")
print("Confusion Matrix")
print(cm,end="\n\n")


#3. SVM Algorithm
from sklearn.svm import SVC
svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)
y_pred3 = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred3)
print("Support Vector Classifier\n")
print("Confusion Matrix")
print(cm,end="\n\n")

#4. Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred4 = nb.predict(X_test)

cm = confusion_matrix(y_test, y_pred4)
print("Gaussian Naive Bays\n")
print("Confusion Matrix")
print(cm,end="\n\n")

from sklearn.naive_bayes import MultinomialNB
nb2 = MultinomialNB()
nb2.fit(x_train, y_train)
y_pred5 = nb2.predict(x_test)

cm = confusion_matrix(y_test, y_pred5)
print("Multinominal Naive Bays\n")
print("Confusion Matrix")
print(cm,end="\n\n")


#decision tree for  classification
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion ="entropy")
dtc.fit(X_train, y_train)
y_pred6=dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred6)
print("Decision Tree\n")
print("Confusion Matrix")
print(cm,end="\n\n")

#random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators = 10,criterion="entropy")
rfc.fit(X_train, y_train)
y_pred7=rfc.predict(X_test)


cm = confusion_matrix(y_test, y_pred7)
print("Random Forest\n")
print("Confusion Matrix")
print(cm,end="\n\n")





