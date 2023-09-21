# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:14:26 2023

@author: Alp Altunsoy
"""

#Associating Rule Mining
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("market_basket.csv", header = None)

t = []


for i in range (0,7501):
    t.append([str(dataset.values[i,j]) for j in range(0,20)])

from apyori import apriori
kurallar = apriori(t, min_support = 0.01, min_confidence = 0.2, min_lift = 3, min_length = 2)
print(list(kurallar))

