# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:41:02 2019

@author: Dell
"""
import math
import numpy as np
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack, coo_matrix

train = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\salary-train.csv')
test = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\salary-test-mini.csv')

train['FullDescription'] = train['FullDescription'].str.lower()
train['LocationNormalized'] = train['LocationNormalized'].str.lower()
train['ContractTime'] = train['ContractTime'].str.lower()

test['FullDescription'] = test['FullDescription'].str.lower()
test['LocationNormalized'] = test['LocationNormalized'].str.lower()
test['ContractTime'] = test['ContractTime'].str.lower()
#train = train.dropna()

train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
train['LocationNormalized'] = train['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)
train['ContractTime'] = train['ContractTime'].replace('[^a-zA-Z0-9]', ' ', regex = True)
train['SalaryNormalized'] = train['SalaryNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)

test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test['LocationNormalized'] = test['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test['ContractTime'] = test['ContractTime'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test['SalaryNormalized'] = test['SalaryNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)


vectorizer = TfidfVectorizer(min_df = 5)
X_train = vectorizer.fit_transform(train['FullDescription'])
X_test = vectorizer.transform(test['FullDescription'])

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)


enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

data = hstack([X_train, X_train_categ])#, coo_matrix(train['SalaryNormalized'].values).transpose()])
data_test = hstack([X_test, X_test_categ])


clf = Ridge(alpha=1.0, random_state=241)
clf.fit(data, train['SalaryNormalized'])

x_prediction = clf.predict(data_test)
print(x_prediction)

a = 0
with open('__first.txt', 'w') as f:
    for i in x_prediction:
        f.write(str(round(i, 2)))
        if a == 0:
            f.write(" ")
            a+=1
#train = train.values

