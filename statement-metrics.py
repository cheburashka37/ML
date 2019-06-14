# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:05:05 2019

@author: Dell
"""

import numpy as np
import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
classif = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\classification.csv')
scores = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\scores.csv')

classif = classif.values

#classif = np.array([[0, 0], [0, 1],  [0, 1], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1]])

metric = -np.array([[0, 0], [0, 0]])
for i in range(len(classif)):
    (pred, true) = (classif[i][0], classif[i][1])
    metric[1 - true][1 - pred]+=1

a = 0
with open('__first.txt', 'w') as f:
    for i in metric:
        f.write(str(i)[1:-1])
        print(str(i)[1:-1])
        if a == 0:
            f.write(" ")
            a+=1


with open('__second.txt', 'w') as f:
    y = classif[:,0]
    x = classif[:,1]
    f.write(str(accuracy_score(y, x)) + ' ' + str(precision_score(y, x)) + ' ' + str(recall_score(y, x)) + ' ' + str(f1_score(y, x)))

print(roc_auc_score(scores[['true']], scores[['score_logreg']]))
print(roc_auc_score(scores[['true']], scores[['score_svm']]))
print(roc_auc_score(scores[['true']], scores[['score_knn']]))
print(roc_auc_score(scores[['true']], scores[['score_tree']]), '\n')


precision, recall, thresholds = precision_recall_curve(scores[['true']], scores[['score_tree']])

