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
@click.option('--pathtodataset', '-d', help='dataset path')
@click.option('--numhiddenneurons', '-n', help='number of hidden units or neurons')
@click.option('--scalefactor', '-s', help='scale factor', type=int)
@click.option('--fillmissingvalues', '-f', help='fill missing values yes (1) or no (0)')
def run(pathtodataset, numhiddenneurons, scalefactor, fillmissingvalues):
    data = np.genfromtxt(sys.stdin if not sys.stdin.isatty() else pathtodataset, dtype=np.float, skip_header=1, delimiter=',', missing_values="")
    train_x, train_y, test_x, test_y = process(data, scalefactor, fillmissingvalues)

    xlm = XLM(int(numhiddenneurons))

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