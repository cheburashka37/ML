# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:35:01 2019

@author: Dell
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
boston = load_boston()
sourse = scale(X = boston.data)
target = boston.target
P = np.linspace(1, 10, 200, endpoint=True)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

max = -200
i = 0
for p in P:
        #neigh = KNeighborsClassifier(n_neighbors=k)
        neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p) 
        a = cross_val_score(X = sourse, y = target, estimator = neigh.fit(sourse, target), scoring='neg_mean_squared_error', cv = kf.split(sourse))
        n = a[0] + a[1] + a[2] + a[3] + a[4]
        print(i, '\t\t')
        i+=1
        if (n > max) :
            max = n
            print(p, ':::', n/5)
        else :
            print('\t\t\t', p, ':::', n/5)
