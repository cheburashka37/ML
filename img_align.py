# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 00:32:52 2019

@author: Dell
"""
import pandas
import cv2 as cv
#from __future__ import division
#import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy.random as npr

def resize(image, width = 64, height = 64):
    dim = (width, height)
     
    # уменьшаем изображение до подготовленных размеров
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized


def bgr2rgb(img):
    b,g,r = cv.split(img)
    return cv.merge([r,g,b])

def rgb2bgr(img):
    b,g,r = cv.split(img)
    return cv.merge([r,g,b])

def get_grey(img):
    w, h, d = img.shape
    img = img.reshape((w*h), d)
    img_new = (0.2989 * img[:,0] + 0.5870 * img[:,1] + 0.1140 * img[:,2]).round().astype(int)
#    img_new = (img[:,0] + img[:,1] + img[:,2])/3
    return img_new.reshape((w,h))

# =============================================================================
# 
# =============================================================================
def convert_y_to_vect(y, nn_structure):
    y_vect = np.zeros((len(y), nn_structure[2]))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


def f(x):
    return 1 / (1 + np.exp(-x))


def f_deriv(x):
    return f(x) * (1 - f(x))


def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = npr.random_sample((nn_structure[l], nn_structure[l-1]))/1000
        b[l] = npr.random_sample((nn_structure[l],))/1000
    return W, b


def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise,
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z


def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)


def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)


def train_nn(nn_structure, X, y, iter_num=300, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        print(cnt)
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y

def show_work(W, b, X, n_layers, i):
    plt.imshow(X_test[i].reshape((8,8)), cmap="gray")
    print(predict_y(W, b, X_test, 3)[i])



data = pandas.read_csv('E:\\User\\Desktop\\Khlamskaya_prog\\IAD\\img_align_celeba\\list_attr_celeba.csv')

df = pandas.DataFrame(data.values[1:], columns=data.values[0])
sourse = df[['img']].values
target = df[['Male']].values

sourse = sourse.reshape(sourse.shape[0])
target = target.reshape(target.shape[0])

for i in df.index :
    #print(i)
    #print(data[i])
    if(target[i] == '1'):
        target[i] = 1
    else :
        target[i] = 0

def prep_sourse(sourse, num = 20, start = 0, w = 64, h = 64):
    X_train = np.zeros((num, w*h))
    for i in range(num):
        name = sourse[start + i]
        img = bgr2rgb(cv.imread('E:\\User\\Desktop\\Khlamskaya_prog\\IAD\\img_align_celeba\\images\\' + name))
        img = resize(img, w, h)
        img = get_grey(img).reshape(-1)
        X_train[i] = img
#        plt.imshow(img)
    return X_train


p_resized_shape = 32
num = 300
start = 100

nn_structure = [p_resized_shape*p_resized_shape, 1000, 2]
# load data and scale

X = prep_sourse(sourse, num = num, start = start, w = p_resized_shape, h = p_resized_shape)
X = X/255
#plt.imshow(X[2].reshape((p_resized_shape, p_resized_shape)), cmap='gray')
target = target[start: start + num]

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.4)
# convert digits to vectors
y_v_train = convert_y_to_vect(y_train, nn_structure)
y_v_test = convert_y_to_vect(y_test, nn_structure)
# setup the NN structure
# train the NN

print(X_train.shape)
W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train, iter_num=100, alpha=0.002)
# plot the avg_cost_func
plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()
# get the prediction accuracy and print
y_pred = predict_y(W, b, X_test, 3)
y_test = np.array(y_test, dtype=int)
print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))
