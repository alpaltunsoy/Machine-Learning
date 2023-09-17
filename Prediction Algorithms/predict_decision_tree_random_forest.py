# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:34:09 2023

@author: Alp Altunsoy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyp

dataset = pd.read_csv("maaslar.csv")

#education level slicing
x = dataset.iloc[:,1:2]
X=x.values
#salaries
y = dataset.iloc[:,2:3]
Y=y.values

#♠decision tree using
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)

#plotting
pyp.title("Decision Tree")
pyp.scatter(X,Y,color ="red",label = "Original values")
pyp.plot(X,r_dt.predict(X),color="blue",label="Decision Tree")
print(r_dt.predict([[6.6]]))

pyp.legend()
pyp.show()










