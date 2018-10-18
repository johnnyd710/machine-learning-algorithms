import numpy as np

class Lasso:

    def __init__(self, w, tol):
        self.w = w
        self.tol = tol

    def no_convergence(self, new_w):
        '''
        no_convergance returns true if the difference between
        the new weight value (new_w) and the old weight value (self.w)
        is less than the tolerance (self.tol)
        updates the weight vector (self.w) otherwise
        '''
        ret_val = True
        if np.abs(new_w - self.w).sum() < self.tol:
            ret_val = False
        else:
            self.w = new_w
        return ret_val

    def soft_threshold(self, wi, li):
        '''
        soft_threshold returns the argmin (1/2)(z-w)^2 + lambda * abs(z)
        '''
        return np.sign(wi)*np.max([0, np.abs(wi)-li])

    def train(self, x, y, l):
        '''
        Algorithm 2: alternating minimization for lasso
        '''
        w = np.zeros(x.shape[1])
        cont = True
        while cont:
            for j in range(0,w.size):
                z = w.copy()
                w[j] = self.soft_threshold( np.dot(x[:,j], y - np.dot(x,z)),
                         l * x.shape[0])/(x[:,j]**2).sum()
            cont = self.no_convergence(w)
        #print(w)

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
            print("\% of non-zero in last 1000 w:", np.count_nonzero(self.w[13:self.w.size])/10)
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
