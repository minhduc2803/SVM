# -*- coding: utf-8 -*-
"""Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1piCJZO0pxU8lzF_K6Ny5THCNxEnylLQi
"""

import numpy as np
import pickle
import gzip
from sklearn import svm
import time

def read_mnist(mnist_file):
    f = gzip.open(mnist_file, 'rb')
    train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    
    train_X, train_Y = train_data
    val_X, val_Y = val_data
    test_X, test_Y = test_data    
    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

def linear_kernel(C):

    clf = svm.SVC(C=C, kernel='linear', cache_size=1000)
    train_time = time.time()
    clf.fit(train_X, train_Y)
        
    train_time = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - train_time))
    E_in = np.mean(clf.predict(train_X) != train_Y)
    E_val = np.mean(clf.predict(val_X) != val_Y)

    print(f'C: {C}')
    print(f'Train error: {E_in}')
    print(f'Validation error: {E_val}')
    print(f'Train time: {train_time}')

def rbf_kernel(C, g):
    
    clf = svm.SVC(C=C, kernel='rbf', gamma=g, cache_size=1000)

    train_time = time.time()
    clf.fit(train_X, train_Y)

    train_time = time.strftime('%Hh:%Mm:%Ss', time.gmtime(time.time() - train_time))
    E_in = np.mean(clf.predict(train_X) != train_Y)
    E_val = np.mean(clf.predict(val_X) != val_Y)

    print(f'C: {C}')
    print(f'gamma: {g}')
    print(f'Train error: {E_in}')
    print(f'Validation error: {E_val}')
    print(f'Train time: {train_time}')

    return clf

train_X, train_Y, val_X, val_Y, test_X, test_Y = read_mnist('mnist.pkl.gz')
C_range = [0.001, 0.01, 0.1, 1, 10, 100]
g_range = [0.001, 0.01, 0.1, 1, 10, 100]

for C in C_range:
    linear_kernel(C)

rbf_kernel(0.001, 0.001)

rbf_kernel(0.001, 0.01)

rbf_kernel(0.001, 0.1)

rbf_kernel(0.001, 1)

rbf_kernel(0.001, 10)

rbf_kernel(0.001, 100)

rbf_kernel(0.01, 0.001)

rbf_kernel(0.01, 0.01)

rbf_kernel(0.01, 0.1)

rbf_kernel(0.01, 1)

rbf_kernel(0.01, 10)

rbf_kernel(0.01, 100)

rbf_kernel(0.1, 0.001)

rbf_kernel(0.1, 0.01)

rbf_kernel(0.1, 0.1)

rbf_kernel(0.1, 1)

rbf_kernel(0.1, 10)

rbf_kernel(0.1, 100)

rbf_kernel(1, 0.001)

rbf_kernel(1, 0.01)

rbf_kernel(1, 0.1)

rbf_kernel(1, 1)

rbf_kernel(1, 10)

rbf_kernel(1, 100)

rbf_kernel(10, 0.001)

rbf_kernel(10, 0.01)

rbf_kernel(10, 0.1)

rbf_kernel(10, 1)

rbf_kernel(10, 10)

rbf_kernel(10, 100)

rbf_kernel(100, 0.001)

rbf_kernel(100, 0.01)

rbf_kernel(100, 0.1)

rbf_kernel(100, 1)

rbf_kernel(100, 10)

rbf_kernel(100, 100)

clf = svm.SVC(C=10, kernel='rbf', gamma=0.01, cache_size=1000)
clf.fit(train_X, train_Y)
E_test = np.mean(clf.predict(test_X) != test_Y)

print(f'Test error: {E_test}')