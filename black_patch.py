# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:13:33 2019

@author: Dell
"""
import pandas
import numpy as np
from skimage import img_as_float
from skimage.io import imread#, imsave
from scipy.misc import imsave
import functions as func

class Const:
    name = property(fget=lambda self: 'Foggy day in Hallstatt')
    patch_size = 15
    per_brightest = 0.0001
    omega = 0.95
    t_min = 0.1
    A_max = 220
    eps = 1e-3
    r = 40

CONST = Const()


image = img_as_float(imread('E:\\User\\Desktop\\Khlamskaya_prog\\IPTI\\' + CONST.name + '.jpg'))
w, h, d = image.shape

w = (w // CONST.patch_size) * CONST.patch_size
h = (h // CONST.patch_size) * CONST.patch_size

image = image[0:w, 0:h,:]

# 2. Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - значениями интенсивности
# в пространстве RGB.
pixels = pandas.DataFrame(np.reshape(image, (w*h, d)), columns=['R', 'G', 'B'])
_w = w//CONST.patch_size
_h = h//CONST.patch_size


radiance = func.get_radiance(image, Const)
#save_radianced_image(radiance, CONST)
#build_histogram(radiance, CONST)
a = func.get_atmosphere_A(image, radiance, CONST)
w,d = radiance.shape
r = radiance.reshape(w*d,1)
r = np.hstack((r, r, r))
r = r.reshape(w, d, 3)
func.save_image(r, CONST)
'''
for i in range(1,10):
    im = boxfilter(np.average(image, axis=2), i)
    imsave('E:\\User\\Desktop\\Khlamskaya_prog\\IPTI\\___boxfilter_' + str(i) + '.jpg', im)
'''