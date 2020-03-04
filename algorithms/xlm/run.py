"""
accepts input from commandline
and runs algorithm,
outputing to command line
"""

from xlm import XLM
import click
import numpy as np
import sys


@click.command()
@click.option('--numhidden', '-n', help='number of hidden units or neurons')
def run(numhidden):
    xlm = XLM(int(numhidden))

    
    train_data, test_data = process()
    train_data_scaled = train_data[:,1:] / 255.0
    xlm.fit(train_data_scaled, train_data[:,0])
    # test_data = np.array(test_data)
    if test_data.shape[1] == train_data.shape[1]:
        labels = test_data[:,0]
        test_data = test_data[:,1:]

    test_data /= 255.0
    predictions = xlm.predict(test_data)
    round_predictions = np.round(predictions, 0)
    if len(labels) > 0:
        print(np.sum(labels == round_predictions, 0) / len(labels))

def process():
    train, test = [], []
    train_mode = True
    first = True
    data = sys.stdin.read().splitlines()
    for line in data:
        if first:
            first = False
            continue

        if train_mode == False:
            test.append([float(x) for x in line.split(',')])

        if (line == '-'):
            train_mode = False

        if (train_mode):
            train.append([float(x) for x in line.split(',')])

    return np.row_stack(train), np.row_stack(test)


if __name__ == "__main__":
    run()