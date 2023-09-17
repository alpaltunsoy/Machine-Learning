# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 20:26:16 2023

@author: Alp Altunsoy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyp

datas = pd.read_csv("datas.csv")

from sklearn import preprocessing

#prep of encoding
label_encoder = preprocessing.LabelEncoder()
one_hat_encoder = preprocessing.OneHotEncoder()

#country encoding
country = datas.iloc[:,0:1]
country= one_hat_encoder.fit_transform(datas.iloc[:,0:1]).toarray()
country = pd.DataFrame(data=country, index = range(22), columns= ["fr","tr", "us"])

#gender encoding
gender = datas.iloc[:,4:5].values
gender = one_hat_encoder.fit_transform(gender).toarray()
male = pd.DataFrame(data = gender[:,0],columns = ["male"], index = range(22))

#data prep
sonuc = datas.iloc[:,1:4].values
sonuc = pd.DataFrame(data = sonuc, index = range(22), columns =["boy", "kilo","yas"])
sonuc = pd.concat([country,sonuc],axis =1)

#preperation of test and train variables for predicting gender
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sonuc, male, test_size = 0.33, random_state = 0)

#importing and setting up modules of lr
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()

#predicting gender
linear_regression.fit(x_train, y_train)
y_predict = linear_regression.predict(x_test)
print("Real Values")
print(y_test)

print("Expected Values")
print(y_predict)


#data prep for height prep
height = pd.DataFrame(data = datas.iloc[:,1:2].values, index = range(22), columns = ["height"])
new_data = pd.concat([sonuc.iloc[:,0:3],sonuc.iloc[:,4:6]],axis=1)
new_data = pd.concat([new_data,male],axis=1)

#preperation of test and train variables for predicting height
x2_train, x2_test, y2_train, y2_test = train_test_split(new_data, height, test_size = 0.33, random_state = 0)

#predicting height
linear_regression2 = LinearRegression()
linear_regression2.fit(x2_train,y2_train)

y2_predict = linear_regression2.predict(x2_test)

print("Real Values")
print(y2_test["height"].values)

print("Expected Values")
print(y2_predict)

