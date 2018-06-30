import os
import random
import numpy as np 
from six.moves import cPickle as pickle
from AlgoBase import AlgoBase
from collections import defaultdict

class NMF(AlgoBase):
    def __init__(self, k, it, biased, reg, reg2, lr, submission, verbose=False):
        self.k = k
        self.it = it
        self.biased = biased
        self.reg = reg
        self.lrf = lr
        self.reg2 = reg2
        self.submission = submission
        self.verbose = verbose

        AlgoBase.__init__(self)

    def train(self):
        split_ratio = 0.90 if not self.submission else 1
        self.split_data(split_ratio)
        predictions = self.sgd()
        #predictions[predictions > 5.0] = 5.0
        #predictions[predictions < 1.0] = 1.0
        #self.predictions = predictions

    def sgd(self):
        users, items = 10000, 1000
        reg2 = self.reg2
        reg = self.reg
        lrf = self.lrf
        global_mean = self.global_mean

        # Randomly initialize user and item factors
        U = np.random.uniform(0, 0.05, (users, self.k)) # pu
        V = np.random.uniform(0, 0.05, (items, self.k)) # qi

        bu = np.zeros(users)
        bi = np.zeros(items)

        if not self.biased:
            global_mean = 0

        if self.verbose:
            print("NMF solving started. Params: k={}, reg={}, reg2={}, lr={}".format(self.k, self.reg, self.reg2, self.lrf))
            print("NMF: There are {} predictions given.".format(len(self.train_data)))
            print("NMF: global mean is {}".format(global_mean))

        ur = defaultdict(list)
        ir = defaultdict(list)

        for u, i, r in self.train_data:
            u, i = u-1, i-1
            ur[u].append(i)
            ir[i].append(u)

        lr = self.nmf_learning_rate(0)
        for current_epoch in range(self.it):
            lr = self.nmf_learning_rate(current_epoch)
            if self.verbose:
                try:
                    score = self.RMSE(self.train_data, U.dot(V.T) + bu.reshape(-1,1) + bi + self.global_mean)
                    test_score = self.RMSE(self.test_data, U.dot(V.T) + bu.reshape(-1,1) + bi + self.global_mean) if self.test_data is not None else -1
                except:
                    print('Error in evaluation')
                else:
                    print("NMF: step {:8d}  ({:2d}% done). fit = {:.4f}, test_fit={:.4f}, learning_rate={:.5f}".format(
                        current_epoch+1, int(100 * (current_epoch+1)/self.it), score, test_score, lr))
                

            # (re)initialize nums and denoms to zero
            user_num = np.zeros((users, self.k))
            user_denom = np.zeros((users, self.k))
            item_num = np.zeros((items, self.k))
            item_denom = np.zeros((items, self.k))

            # Compute numerators and denominators for users and items factors
            for u, i, r in self.train_data:
                u,i = u-1,i-1
                # compute current estimation and error
                dot = 0  # <q_i, p_u>
                for f in range(self.k):
                    dot += V[i, f] * U[u, f]
                est = global_mean + bu[u] + bi[i] + dot
                err = r - est

                # update biases
                if self.biased:
                    bu[u] += lr * (err - reg2 * bu[u])
                    bi[i] += lr * (err - reg2 * bi[i])
                    #bu[u] += lr * (err - reg2 * (bu[u] + bu[i] - global_mean))
                    #bi[i] += lr * (err - reg2 * (bi[i] + bu[u] - global_mean))

                # compute numerators and denominators
                for f in range(self.k):
                    user_num[u, f] += V[i, f] * r
                    user_denom[u, f] += V[i, f] * est
                    item_num[i, f] += U[u, f] * r
                    item_denom[i, f] += U[u, f] * est

            # Update user factors
            for u, _, _ in self.train_data:
                u = u-1
                n_ratings = len(ur[u])
                for f in range(self.k):
                    user_denom[u, f] += n_ratings * reg * U[u, f]
                    if user_denom[u, f] != 0:
                        U[u, f] *= user_num[u, f] / user_denom[u, f]
                    else:
                        print('Div by zero avoided in users')

            # Update item factors
            for _, i, _ in self.train_data:
                i = i - 1
                n_ratings = len(ir[i])
                for f in range(self.k):
                    item_denom[i, f] += n_ratings * reg * V[i, f]
                    if item_denom[i, f] != 0:
                        V[i, f] *= item_num[i, f] / item_denom[i, f]
                    else:
                        print('Div by zero avoided in items')

        self.bu = bu
        self.bi = bi
        self.U = U
        self.V = V
        #return U.dot(V.T) + bu.reshape(-1,1) + bv + self.global_mean

    def nmf_learning_rate(self, s):
        """
        Adjusting rate.
    
        s : iteration counter
        """
        #rates = [0.035, 0.032, 0.029, 0.027, 0.012, 0.01, 0.0022, 0.002, 0.00055, 0.0005, 0.0001, 0.00002]
        rates = [0.0033, 0.0030, 0.0029, 0.0025, 0.0022, 0.0020, 0.0017,0.0015, 0.0013, 0.0011, 0.0009, 0.0007,
         0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001,0.000085, 0.00007, 0.00005, 0.00004, 0.00003, 
         0.00002,0.000015, 0.00001, 0.000007, 0.000005, 0.000003, 0.000002, 0.000001]
        progress = round((s / self.it) * len(rates))
        return rates[progress] * self.lrf

    def predict(self, u, i, verbose=False):
        if self.biased:
            est = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.V[i], self.U[u])
        else:
            est = np.dot(self.V[i], self.U[u])
        return est

sgd = NMF(8, 20, True, 0.08, 0.04, 2.5, True, True)
sgd.train()
sgd.generatePredictions('../data/NMF1.csv')