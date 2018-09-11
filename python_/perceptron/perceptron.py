'''
Perceptron Algorithm developed in 1958 by Rosenblatt
'''
import numpy as np
import sys
import time
from numpy import genfromtxt

def fetch_data(filename):
    data = genfromtxt(filename, delimiter=',', skip_header=True)
    return data

def Perceptron(D):
    w = np.zeros(np.size(D,1)-1)
    b = 0
    threshold = 0
    while True:
        for instance in D:
            x = instance[:-1]
            y = instance[-1]
            if y*(np.dot(w,x) + b) <= threshold:
                w += y*x
                b += y
            print("Weight Vector:", w)
            time.sleep(1)

filename = sys.argv[1]
D = fetch_data(filename)
if np.size(D) != 0:
    Perceptron(D)
else:
    print("Data empty or wrong filename")