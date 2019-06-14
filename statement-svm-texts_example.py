# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:39:00 2019

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn import datasets
newsgroups=datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
X=vectorizer.fit_transform(newsgroups.data)
y=newsgroups.target
from sklearn.svm import SVC
from sklearn.model_selection import KFold
kf=KFold(n_splits=5, shuffle=True, random_state=241)
from sklearn.model_selection import GridSearchCV
grid={'C':np.power(10.0,np.arange(-5,6))}
clf=SVC(kernel='linear',random_state=241)
gs=GridSearchCV(clf,grid, scoring='accuracy',cv=kf)
gs.fit(X,y)
for a in gs.grid_scores_:
    print(a.mean_validation_score)
    print(a.parameters)
clf2=SVC(kernel='linear',C=1.0, random_state=241)
clf2.fit(X,y)
coef=clf2.coef_
q=pd.DataFrame(coef.toarray()).transpose()
top10=abs(q).sort_values([0], ascending=False).head(10)
indices=[]
indices=top10.index
words=[]
for i in indices:
    feature_mapping=vectorizer.get_feature_names()
    words.append(feature_mapping[i])
print(sorted(words))