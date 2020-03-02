'''
Question 1 Implement Ridge Regression
'''

import numpy as np

class Ridge_regression:

    def __init__(self, l, w):
        self.l = l
        self.w = w

    def no_convergence(self):
        return True

    def train(self, x, y, l):
        Xt = np.transpose(x)
        lambda_identity = l*np.identity(np.size(x,1))
        #theInverse = np.linalg.inv(np.dot(Xt, self.X)+lambda_identity)
        #w = np.dot(np.dot(theInverse, Xt), self.Y)
        A = np.dot(Xt,x) + lambda_identity
        #y.shape=(,1)
        b = np.dot(Xt,y)
        self.w = np.linalg.solve(A, b)

    def cross_validate(self, trainx, trainy, testx, testy, num_folds=10):
        '''
        cross validation for num_folds on trainx and trainy
        and prints performance on test set (testx, testy)
        '''
        l_ = []
        perf = []
        incr = int(np.size(trainy)/num_folds)
        for l in range(0,110,10):
            print("Lambda ", l)
            start = 0
            l_.append(l)
            p = []
            for fold in range(0,num_folds):
                x = np.delete(trainx, range(start,(start+incr)), 0)
                y = np.delete(trainy, range(start,(start+incr)))
                self.train(x,y,l)
                p.append(self.test(trainx[start:(start+incr)],
                            trainy[start:(start+incr)]))
                start += incr
            perf.append(np.mean(p))
            print("Validation", np.mean(p))
            self.train(trainx, trainy, l)
            print("Training", self.test(trainx, trainy))
            print("Testing", self.test(testx, testy))
            print("\% of non-zero in w:", np.count_nonzero(self.w)/self.w.size*100)
            print("\n")

        self.l = l_[np.argmin(perf)]
        #self.train(self.X, self.Y, self.l)
        #print("Best lambda: ", self.l)

    def test(self, X_test, Y_test):
        '''
        evaluates mean squared error on prediction
        '''
        n = Y_test.size
        f = np.dot(X_test, self.w)
        return (np.linalg.norm(f - Y_test)**2) / n
