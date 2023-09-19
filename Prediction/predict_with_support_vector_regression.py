# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:34:09 2023

@author: Alp Altunsoy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyp
from sklearn.metrics import r2_score

dataset = pd.read_csv("maaslar.csv")

#education level
x = dataset.iloc[:,1:2]
#salaries
y = dataset.iloc[:,2:3]
Y=y.values

#train and test variable declaration
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size =0.33)

#standart linear model
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()

#predicting with standart linear regression
linear_regression.fit(x,y)
y_predict_with_standart_lr = linear_regression.predict(x)

#plotting
pyp.scatter(x,y,label="Salaries",color="red")
pyp.plot(x ,y_predict_with_standart_lr, label = "Standart Linear Regression")


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
#♣creating poly with degree
poly_reg=PolynomialFeatures(degree=3)
#creating x values as x_poly
x_poly = poly_reg.fit_transform(x)

linear_regression2 = LinearRegression()
#creating new linear model with polynomial x values
linear_regression2.fit(x_poly,y)
y_predict_with_polynomial_linear = linear_regression2.predict(x_poly)

#plotting
pyp.plot(x ,y_predict_with_polynomial_linear, label = "Polinomial Linear Regression")
pyp.legend()
pyp.show()


from sklearn.preprocessing import StandardScaler

scl= StandardScaler()
x_scaled = scl.fit_transform(x)

scl2= StandardScaler()
y_scaled=np.ravel(scl2.fit_transform(Y.reshape(-1,1)))

#SVM
from sklearn.svm import SVR

svr_regression =SVR(kernel ="rbf")
svr_regression.fit(x_scaled, y_scaled)

pyp.scatter(x_scaled,y_scaled)
pyp.plot(x_scaled,svr_regression.predict(x_scaled))
pyp.legend()
pyp.show()
print("Svm r2_score value: ")
print(r2_score(y_scaled, svr_regression.predict(x_scaled)))












