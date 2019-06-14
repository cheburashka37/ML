# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:53:09 2019

@author: Dell
"""

import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold

data = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

y = data[['Rings']]
X = data[['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']]

X = X.values
y = y.values

#print(r2_score([10, 11, 12], [9, 11, 12.1]))
kf = KFold(n = len(data), n_folds =5, shuffle=True, random_state=1)


for i in range(1, 51):
    n = 0;
    for train_index, test_index in kf:
        clf = RandomForestRegressor(n_estimators=i, random_state=1)
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        a = clf.fit(X_train, y_train[:,0])
        predictions = clf.predict(X_test)
        n += r2_score(y_true = y_test, y_pred = predictions)
    
    n = n / 5
    if (n > 0.52):
        print(i, ':::', n)