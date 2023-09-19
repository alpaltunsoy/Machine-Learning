# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:42:41 2023

@author: Alp Altunsoy
"""

#modules that we will use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#taking dataset
original_dataset = pd.read_csv("maaslar_detayli.csv")

#slicing we want to predict salary so dependent variable is salary
x = original_dataset.iloc[:,2:5]
x_temp =x.values
y = original_dataset[["maas"]].values

#Linear regression 
from sklearn.linear_model import LinearRegression

lr =LinearRegression()
lr.fit(x,y)
lr_predict = lr.predict(x_temp)

#make a decision for variables 
import statsmodels.api as sm 

model = sm.OLS(endog=lr_predict, exog = x)
print(model.fit().summary()) #kidem is higher we have to remove from x

#removing kidem
x_temp = pd.concat([x.iloc[:,0:1],x.iloc[:,2:3]],axis = 1)


#again try for make a decision
model = sm.OLS(endog= lr_predict, exog=x_temp)

print(model.fit().summary()) #point is very much remove 


#removing point from x_temp
x_temp = x_temp[["UnvanSeviyesi"]] 

#again try for make a decision
model = sm.OLS(endog= lr_predict, exog=x_temp)
print("Linear Results")
print(model.fit().summary())

#Again Linear Regression
lr2 = LinearRegression()
lr2.fit(x_temp, y)
lr_predict = lr2.predict(x_temp)



print("*"*1455)



#predict with poly regression
x_temp = x_temp.values
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 4)
poly_x = poly.fit_transform(x_temp)

poly_lr = LinearRegression()
poly_lr.fit(poly_x,y)
poly_y_prediction = poly_lr.predict(poly_x)

#taking results
model2 = sm.OLS(endog = poly_y_prediction, exog = x_temp)
print("Polynomial Results")
print(model2.fit().summary())
print("*"*1455)

#predict with SVM

from sklearn.preprocessing import StandardScaler

scl= StandardScaler()
x_scaled = scl.fit_transform(x_temp)

scl2= StandardScaler()
y_scaled= np.ravel(scl2.fit_transform(y.reshape(-1,1)))

from sklearn.svm import SVR

svr_regression =SVR(kernel ="rbf")
svr_regression.fit(x_scaled, y_scaled)


#taking results from svm prediction
model3 = sm.OLS(endog = svr_regression.predict(x_scaled), exog = x_scaled)
print("SVM results")
print(model3.fit().summary())
print("*"*1455)

#decision tree prediction
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state = 0)
tree.fit(x_temp, y)
dt_prediction = tree.predict(x_temp)

print("DT results")
model4 = sm.OLS(endog=dt_prediction, exog = x_temp)
print(model4.fit().summary())

#random forest prediction 

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=0, n_estimators = 10)
rfr.fit(x_temp, y)
rfr_prediction = rfr.predict(x_temp)

print("RFR results")
model5 = sm.OLS(endog=rfr_prediction, exog = x_temp)
print(model5.fit().summary())


#plotting

plt.plot(y,color="black",label="Original Values")
plt.plot(lr_predict,color="red",label="Linear Regression")
plt.legend()
plt.show()

plt.plot(y,color="black",label="Original Values")
plt.plot(poly_y_prediction,color="blue",label="Poly Linear Regression")
plt.legend()
plt.show()

plt.plot(svr_regression.predict(x_scaled),color="green",label="SVR ")
plt.legend()
plt.show()

plt.plot(y,color="black",label="Original Values")
plt.plot(dt_prediction,color="yellow",label="DT")
plt.legend()
plt.show()

plt.plot(y,color="black",label="Original Values")
plt.plot(rfr_prediction,color="yellow",label="RFR")
plt.legend()
plt.show()










