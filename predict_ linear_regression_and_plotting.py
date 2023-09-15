# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:08:51 2023

@author: Alp Altunsoy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing datas
datas = pd.read_csv("sales.csv")
print(datas)

months  = datas[["Aylar"]]
sales  = datas[["Satislar"]]

print(months)
print(sales)


from sklearn.model_selection import train_test_split
#aylar bağımsız satışlar bağımlı değişken ona göre yerleştirmek önemli
x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size = 0.33)

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

#standardalize ettik aynı dünyaya indirgemek
X_train = standard_scaler.fit_transform(x_train) 
X_test = standard_scaler.fit_transform(x_test)

Y_train = standard_scaler.fit_transform(y_train) 
Y_test = standard_scaler.fit_transform(y_test) 

#model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, Y_train)
predict = lr.predict(X_test)

lr.fit(x_train, y_train)
predict_2 = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train) 
plt.plot(x_test, predict_2)

plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.title("Satış tahimini")
plt.show()

