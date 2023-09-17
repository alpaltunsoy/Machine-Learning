# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 23:10:03 2023

@author: Alp Altunsoy
"""

#importing modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyp


#import dataset
dataset = pd.read_csv("tennis_dataset.csv")

#set up for encoding
from sklearn import preprocessing

one_hat_encode = preprocessing.OneHotEncoder()
label_encode = preprocessing.LabelEncoder()

#windy encode
windy = dataset[["windy"]].values
windy = pd.DataFrame(data=label_encode.fit_transform(windy[:,0]),columns = ["windy"])

#play encode 
play = dataset[["play"]].values
play = pd.DataFrame(data= label_encode.fit_transform(play[:,0]),columns = ["play"])

#outlook encode
outlook = dataset.iloc[:,0:1].values
outlook = pd.DataFrame(data = one_hat_encode.fit_transform(dataset.iloc[:,0:1]).toarray(),columns =["overcast","rainy","sunny"] )

#combine dataFrames
new_data = pd.concat([outlook, dataset[["temperature", "humidity"]]],axis=1)
new_data = pd.concat([new_data,windy],axis=1)

#creating models for first try
from sklearn.model_selection import  train_test_split
x1_train,x1_test,y1_train,y1_test = train_test_split(new_data, play, test_size=0.33,random_state =0)

#first attempt for predict basic multiple regression
from sklearn.linear_model import LinearRegression
linear_regression_1 = LinearRegression()

linear_regression_1.fit(x1_train, y1_train)
y1_predict = linear_regression_1.predict(x1_test)

#backward elimination 

import statsmodels.api as sm

#beta 0 adding

beta0 = np.append(arr = np.ones((14,1)).astype(int), values = new_data,axis=1)

variable_list= new_data.iloc[:,[0,1,2,3,4,5]].values
variable_list = np.array(variable_list,dtype=float)

model = sm.OLS(play,variable_list).fit()
print(model.summary())
#x4 highest 


#x4 removed
variable_list= new_data.iloc[:,[0,1,2,4,5]].values
variable_list = np.array(variable_list,dtype=float)

model = sm.OLS(play,variable_list).fit()
print(model.summary())
#again x4 is higher

#x4 removed
variable_list= new_data.iloc[:,[0,1,2,5]].values
variable_list = np.array(variable_list,dtype=float)

model = sm.OLS(play,variable_list).fit()
print(model.summary())
#again x4 is higher

#x4 removed
variable_list= new_data.iloc[:,[0,1,2]].values
variable_list = np.array(variable_list,dtype=float)

model = sm.OLS(play,variable_list).fit()
print(model.summary())
#x3 is higher

#x3 removed
variable_list= new_data.iloc[:,[0,1]].values
variable_list = np.array(variable_list,dtype=float)

model = sm.OLS(play,variable_list).fit()
print(model.summary())
#x3 is higher

#again linear regression

x2_train,x2_test,y2_train,y2_test = train_test_split( new_data.iloc[:,[0,1]], play, test_size=0.33)
linear_regression_2 = LinearRegression()

linear_regression_2.fit(x2_train, y2_train)
y2_predict = linear_regression_2.predict(x2_test)














