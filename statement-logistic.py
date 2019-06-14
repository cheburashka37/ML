# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:49:00 2019

@author: Dell
"""

import numpy as np
import pandas
import math
from sklearn.metrics import roc_auc_score
data = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\data-logistic.csv')

#data.loc[i, 'Sex']

X = data[['b', 'c']]
X = X.values
y = data['a']
y = y.values

eps = 1e-5

#X = X[1:][:]
#y = y[1:][:]

w1 = 0
w1_old = 0
w2 = 0
w2_old = 0

k = 0.1
l = len(y)
with open('first.txt', 'w') as f:
    
    C = 0
    for iteration in range(int(1e3)):
        Sum1 = 0
        Sum2 = 0
        for i in range(l):
            exp = (1-1/(1 + math.exp(-y[i]*(w1*X[i][0] + w2*X[i][1]))))
            Sum1+= X[i][0]*y[i]*exp 
            Sum2+= X[i][1]*y[i]*exp
        
        #print('\n', iteration, ':')
        #print('Sum1 = ', Sum1, ';   Sum2 = ', Sum2)
        w1_old = w1
        w2_old = w2
        
        w1 = w1 + k*Sum1/l - k*C*w1
        w2 = w2 + k*Sum2/l - k*C*w2
        if math.sqrt((w1 - w1_old)**2 + (w2 - w2_old)**2) < eps:
            break
        #print('w1 = ', w1, ';   w2 = ', w2)
    
    
    a = np.zeros(l)
    for i in range(l):
        a[i] = 1 / (1 + math.exp(-w1*X[i][0] - w2*X[i][1]))
    
    print(round(roc_auc_score(y, a), 3))
    f.write(str(round(roc_auc_score(y, a), 3)))
    
    print('==============================================================')
    
    
    C = 10
    
    for iteration in range(int(1e3)):
        Sum1 = 0
        Sum2 = 0
        for i in range(l):
            exp = (1-1/(1 + math.exp(-y[i]*(w1*X[i][0] + w2*X[i][1]))))
            Sum1+= X[i][0]*y[i]*exp 
            Sum2+= X[i][1]*y[i]*exp
        
        #print('\n', iteration, ':')
        #print('Sum1 = ', Sum1, ';   Sum2 = ', Sum2)
        w1_old = w1
        w2_old = w2
        
        w1 = w1 + k*Sum1/l - k*C*w1
        w2 = w2 + k*Sum2/l - k*C*w2
        if math.sqrt((w1 - w1_old)**2 + (w2 - w2_old)**2) < eps:
            break
        #print('w1 = ', w1, ';   w2 = ', w2)
    
    
    a = np.zeros(l)
    for i in range(l):
        a[i] = 1 / (1 + math.exp(-w1*X[i][0] - w2*X[i][1]))
    
    print(round(roc_auc_score(y, a), 3))
    f.write(str(round(roc_auc_score(y, a), 3)))
    f.write(" ")
    
    
    #w1 =  0.12332762198512852 ;   w2 =  0.08680287094233427
    
    
    #w1 =  0.2881081945730924 ;   w2 =  0.0917091004796382