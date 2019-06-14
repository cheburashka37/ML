# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:51:25 2019

@author: Dell
"""

from sklearn.decomposition import PCA
import pandas
import numpy as np

train = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\close_prices.csv')
test = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\djia_index.csv')

pca = PCA()
pca.fit(train[['AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
       'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
       'NKE', 'PFE', 'PG', 'T', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM']])
print(pca.explained_variance_ratio_)
counter = 0
for i in pca.explained_variance_ratio_:
    counter+=i
    if counter >= 0.9:
        break

X = train[['AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS',
       'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
       'NKE', 'PFE', 'PG', 'T', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM']]
X = pca.transform(X) 

a = np.corrcoef(X[:,0], np.array(test[['^DJI']])[:,0])


with open('__first.txt', 'w') as f:
    f.write(str('UTX'))

max = -10
i_saved = -1
for i in range(0,30):
    #print(X[:,i].shape)
    a = np.corrcoef(X[:,i], np.array(test[['^DJI']])[:,0])
    #print(a[1][0])
    if max < a[1][0]:
        max = a[1][0]
        i_saved = i