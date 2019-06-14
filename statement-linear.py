# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:51:27 2019

@author: Dell
"""

from sklearn.linear_model import Perceptron
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\perceptron-train.csv')
test = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\perceptron-test.csv')

train_source = train[['b', 'c']]
train_target = train[['a']]
test_source = test[['b', 'c']]
test_target = test[['a']]


clf = Perceptron(random_state=241)
clf.fit(train_source, train_target)
predictions = clf.predict(test_source)

accuracy_1 = accuracy_score(test_target, predictions)



scaler = StandardScaler()
train_source = scaler.fit_transform(train_source)
test_source = scaler.transform(test_source)

train_source = pandas.DataFrame(train_source,columns=['a', 'b'])
test_source = pandas.DataFrame(test_source,columns=['a', 'b'])
#train_target_scaled = np.array(range(0, 300))
#for i in range(0, 300):
#    train_target_scaled[i] = train_target[i][0]
#test_source = test_scaled[:,1:3]
#test_target = test_scaled[:,0:1]
#test_target_scaled = np.array(range(0, 200))
#test_target_scaled = pandas.DataFrame(test_target_scaled ,columns=['a'])

#for i in range(0, 200):
#    test_target_scaled[i] = test_target[i][0]



clf = Perceptron(random_state=241)
clf.fit(train_source, train_target)
predictions = clf.predict(test_source)

print(round(-accuracy_1 + accuracy_score(test_target, predictions), 5))
