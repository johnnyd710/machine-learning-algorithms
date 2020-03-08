#!/usr/bin/env python

"""
accepts input from commandline
and runs algorithm,
outputing to command line
"""

from xlm import XLM
import click
import numpy as np
import sys
from process import process


@click.command()
@click.option('--numhidden', '-n', help='number of hidden units or neurons')
def run(numhidden):

    data = np.loadtxt(sys.stdin, dtype=np.int, skiprows=1, delimiter=',')
    train_x, train_y, test_x, test_y = process(data)

    xlm = XLM(int(numhidden))

    xlm.fit(train_x, train_y)
    
    y = xlm.predict(test_x)
    correct = 0
    total = y.shape[0]
    for i in range(total):
        predicted = np.argmax(y[i])
        test = np.argmax(test_y[i])
        correct = correct + (1 if predicted == test else 0)
    # print('Accuracy: {:f}'.format(correct/total))

    np.savetxt(sys.stdout.buffer, y, fmt='%.4f')
    np.savetxt(sys.stdout.buffer, test_y, fmt='%.1f')


if __name__ == "__main__":
    run()