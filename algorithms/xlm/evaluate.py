import numpy as np
import sys

data = np.loadtxt(sys.stdin, delimiter=' ')
length = len(data)
predicted, actual = np.split(data, 2)

correct = 0
total = predicted.shape[0]
conf_matrix = np.zeros((predicted.shape[1], predicted.shape[1]), dtype=int)
for i in range(total):
    predict = np.argmax(predicted[i])
    test = np.argmax(actual[i])
    correct = correct + (1 if predict == test else 0)
    conf_matrix[predict-1, test-1] += 1

print('Accuracy: {:f}'.format(correct/total))
# print(conf_matrix)

true_pos = np.diag(conf_matrix)
false_pos = np.sum(conf_matrix, axis=0) - true_pos
false_neg = np.sum(conf_matrix, axis=1) - true_pos

precision = np.sum(true_pos / (true_pos + false_pos))
recall = np.sum(true_pos / (true_pos + false_neg))

print('Precision: {:f}'.format(precision))
print('Recall: {:f}'.format(recall))
