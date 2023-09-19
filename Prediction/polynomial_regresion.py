# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:34:09 2023

@author: Alp Altunsoy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyp

dataset = pd.read_csv("salaries.csv")

#slicing
x = dataset.iloc[:,1:2]
y = dataset.iloc[:,2:3]

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