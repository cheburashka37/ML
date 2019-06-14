# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:25:22 2019

@author: Dell
"""

import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
#print(vectorizer.get_feature_names())
#['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
#print(X.shape)
#(4, 9)

'''
feature_mapping = vectorizer.get_feature_names()
for i in range(1, 10):
    print(i, ':::', feature_mapping[1])
'''

grid = {'C': np.power(10.0, np.arange(-5, 6))}

cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

clf.fit(X, y)

clf.coef_

a = np.array(clf.coef_.data)
top = sorted(range(len(a)), key=lambda x: math.fabs(a[x]))[-10:]
top.sort()
with open('first.txt', 'w') as f:
    for i in top:
        f.write(vectorizer.get_feature_names()[i])
        if(i != len(clf.support_) - 1):
            f.write(", ")

'''for C in np.power(10.0, np.arange(-5, 6)):
        neigh = KNeighborsClassifier(n_neighbors=k)
        a = cross_val_score(X = sourse, y = target, estimator = neigh.fit(sourse, target), cv = kf.split(sourse), scoring = 'accuracy')
        n = a[0] + a[1] + a[2] + a[3] + a[4]
        if (n > max) :
            max = n
            print(k, ':::', n/5)
        else :
            print('\t\t\t', k, ':::', n/5)
'''