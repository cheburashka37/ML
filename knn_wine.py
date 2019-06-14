# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 23:19:28 2019

@author: Dell
"""

import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
data = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\wine.data')



X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])

sourse = data[[' Alcohol', ' Malic acid', ' Ash', ' Alcalinity of ash',
       ' Magnesium', ' Total phenols', ' Flavanoids', ' Nonflavanoid phenols',
       ' Proanthocyanins', ' Color intensity', ' Hue',
       ' OD280/OD315 of diluted wines', ' Proline']]
target = data['Class']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
#kf.get_n_splits(X)
#print(kf)  
'''
KFold(n_splits=10, shuffle=False)
for k in range(1, 51):
    for train_index, test_index in kf.split(sourse):
        #print(repr(train_index))
        print("TRAIN:", train_index, "TEST:", test_index)
        sourse_train = sourse.loc[train_index]
        sourse_test = sourse.loc[test_index]
        target_train, target_test = target.loc[train_index], target.loc[test_index]
        
        neigh = KNeighborsClassifier(n_neighbors=k)
        
        
        print(cross_val_score(X = sourse_test, y = target_test, estimator = neigh.fit(sourse_train, target_train), cv = 5, scoring = 'accuracy') )
'''
sourse = scale(X = sourse)
max = 0
#KFold(n_splits=10, shuffle=False)
for k in range(1, 51):
        neigh = KNeighborsClassifier(n_neighbors=k)
        a = cross_val_score(X = sourse, y = target, estimator = neigh.fit(sourse, target), cv = kf.split(sourse), scoring = 'accuracy')
        n = a[0] + a[1] + a[2] + a[3] + a[4]
        if (n > max) :
            max = n
            print(k, ':::', n/5)
        else :
            print('\t\t\t', k, ':::', n/5)