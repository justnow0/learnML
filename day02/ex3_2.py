import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat 
from scipy.optimize import minimize

def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

def load_data(path):
    data = loadmat(path)
    x = data['X']
    y = data['y']
    return x, y

theta1, theta2 = load_weight('ex3weights.mat')
"""
print(theta1.shape, theta2.shape)
(25, 401) (10, 26)
"""
x, y = load_data('ex3data1.mat')
y = y.flatten()
x = np.insert(x, 0, values=np.ones(x.shape[0]),axis=1)
"""
print(x.shape, y.shape)
(5000, 401) (5000,)
"""
a1 = x

z2 = a1 @ theta1.T
z2.shape
"""
print(z2.shape)
(5000, 25)
"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z2 = np.insert(z2, 0, 1, axis=1)
a2 = sigmoid(z2)
"""
print(a2.shape)
(5000, 26)
"""
z3 = a2 @ theta2.T
a3 = sigmoid(z3)
y_pred = np.argmax(a3, axis=1) + 1 
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))  # accuracy = 97.52%


