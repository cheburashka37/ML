# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 23:37:15 2019

@author: Dell
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from scipy.sparse import hstack
import math
import pandas
import numpy as np

data = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\gbm-data.csv')

data = data.values

X = data[:,1:]
y = data[:,0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
'''
min = 2
a = 0
#for learning_rate in [1, 0.5, 0.3, 0.2, 0.1, 0.05]:
    
learning_rate = 0.2
gbc = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate = learning_rate)
gbc.fit(X_train, y_train)

y_pred_gen = gbc.staged_decision_function(X_test)
#print(gbc.staged_decision_function(hstack(X_train, y_train)))
n = 0
test_loss = np.array([])
train_loss = np.array([])

n = 0
for i in y_pred_gen:
    #print(i)
    n+=1
    y_pred = 1 / (1 + np.exp(-i))
    log = log_loss(y_test, y_pred)
    if (log < min):
        min = log
        a = (n, learning_rate)
    test_loss = np.hstack((test_loss, np.array([log])))

y_pred_gen = gbc.staged_decision_function(X_train)
for i in y_pred_gen:
    #print(i)
    y_pred = 1 / (1 + np.exp(-i))
    train_loss = np.hstack((train_loss, np.array([log_loss(y_train, y_pred)])))






import matplotlib.pyplot as plt
#%matplotlib inline
plt.figure()
plt.plot(test_loss, 'r', linewidth=2)
plt.plot(train_loss, 'g', linewidth=2)
plt.legend(['test', 'train'])
'''

clf = RandomForestClassifier(n_estimators=37, random_state=241)

clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)
y_pred = 1 / (1 + np.exp(-predictions))
print(log_loss(y_test, predictions))

