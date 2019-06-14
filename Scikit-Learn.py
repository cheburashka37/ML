# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:43:20 2019

@author: Dell
"""

import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier
data = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\titanic.csv')
X = np.array([[1,2],[3,4],[5,6]])
y = np.array([0,1,0])
#importances = clf.feature_importances_
for i in data.index :
    #print(i)
    #print(data[i])
    if(data.loc[i, 'Sex'] == 'male'):
        data.loc[i, 'Sex'] = 1
    else :
        data.loc[i, 'Sex'] = 0

data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
data = data.dropna()

sourse = data[['Pclass', 'Fare', 'Age', 'Sex']]
target = data['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(sourse, target)
importances = clf.feature_importances_
importances.sort
print(importances[0:2:])