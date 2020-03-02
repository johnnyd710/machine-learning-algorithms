'''
Perceptron Algorithm developed in 1958 by Rosenblatt
ex.
python3 perceptron data.txt "1,1,0,1,1"
'''
import numpy as np
import sys
import time
from numpy import genfromtxt

def fetch_data(filename):
    data = genfromtxt(filename, delimiter=',', skip_header=True)
    return data

class Perceptron:
    def __init__(self, D):
        self.w = np.zeros(np.size(D,1)-1)
        self.b = 0
        self.threshold = 0

    def run(self,D,iter):
        for i in range(iter):
            for instance in D:
                x = instance[:-1]
                y = instance[-1]
                if y*(np.dot(self.w,x) + self.b) <= self.threshold:
                    self.w += y*x
                    self.b += y
                print("Weight Vector:", self.w)
                time.sleep(1)

    def predict(self, x):
        return np.dot(self.w,x) + self.b

filename = sys.argv[1]
new_data = np.asarray(sys.argv[2].split(','), dtype = float)
print(new_data)
D = fetch_data(filename)
if np.size(D) > 1:
    P = Perceptron(D)
    P.run(D, 1)
    y = P.predict(new_data)
    print("Prediction:", y)

else:
    print("Data empty or wrong filename")
