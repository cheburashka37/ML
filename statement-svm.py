# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:59:55 2019

@author: Dell
"""

from sklearn.svm import SVC
import pandas

data = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\svm-data.csv')

source = data[['b', 'c']]
target = data[['a']]

clf = SVC(C = 100000, random_state=241,  kernel='linear')
clf.fit(source, target)


with open('first.txt', 'w') as f:
    for i in range(len(clf.support_)):
        f.write(str(clf.support_[i] + 1))
        if(i != len(clf.support_) - 1):
            f.write(" ")