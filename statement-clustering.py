# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:31:51 2019

@author: Dell
"""

from sklearn.cluster import KMeans
from skimage.io import imread
import numpy as np
import pandas
from skimage import img_as_float
import pylab

image = imread('E:\\User\\Desktop\\Khlamskaya_prog\\Coursera\\MashineLearning\\parrots.jpg')

image = img_as_float(image)
#pylab.imshow(image)


w, h, d = image.shape

# 2. Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - значениями интенсивности
# в пространстве RGB.

pixels = pandas.DataFrame(np.reshape(image, (w*h, d)), columns=['R', 'G', 'B'])


model = KMeans(n_clusters=2, init='k-means++', random_state=241)
pixels['cluster'] = model.fit_predict(pixels)

means = pixels.groupby('cluster').mean().values
mean_pixels = [means[c] for c in pixels['cluster'].values]
