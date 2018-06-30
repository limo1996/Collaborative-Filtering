import os
import random
import numpy as np 
from six.moves import cPickle as pickle
from AlgoBase import AlgoBase

class SGD(AlgoBase):
    """
    Collaborative filtering algorithm for prediction of user item preferences.

    Parameters:
        out_path: Path for saving U, V computed matrices.
        reg : regularizer for the U and V matrices
        reg2 : regularizer for biasU and biasV matrices
        k : the number of features to use
        it: number of iterations 
        lr: learning rate factor
        submission: if true than all data are train data otherwise 80%
    """

    def __init__(self, out_path, reg, reg2, k, it, lr, submission):
        self.reg = reg
        self.reg2 = reg2
        self.out_path = out_path
        self.k = k
        self.it = it
        self.lrf = lr
        self.submission = submission
        AlgoBase.__init__(self)

    def train(self):
        split_ratio = 0.90 if not self.submission else 1
        self.split_data(split_ratio)
        predictions = self.sgd(True)
        predictions[predictions > 5.0] = 5.0
        predictions[predictions < 1.0] = 1.0
        self.predictions = predictions

    def predict(self, user, item, verbose=False):
        """ Predicts rating for (user, item) pair """
        return self.predictions[user, item]

    def sgd(self, verbose):
        """
        Stochastic Gradient Descent (SGD) predictor.
            
        Training set is list of tuples where each tuple contains item id, user id and rating
            
        k : the number of features to use
        reg : regularizer for the U and V matrices
        reg2 : regularizer for biasU and biasV matrices
            
        verbose = if true process is reported
                    
        """
        reg = self.reg
        reg2 = self.reg2

        if os.path.exists(self.out_path):
            U, V, biasU, biasV = pickle.load(open(self.out_path, 'rb'))
        else:
            global_mean = self.global_mean
            users, items = 10000, 1000
            
            U = np.random.uniform(0, 0.05, (users, self.k))
            V = np.random.uniform(0, 0.05, (items, self.k))
        
            biasU = np.zeros(users)
            biasV = np.zeros(items)
                
            if verbose:
                print("SGD solving started. Params: k={}, reg={}, reg2={}, lr={}".format(self.k, self.reg, self.reg2, self.lrf))
                print("SGD: There are {} predictions given.".format(len(self.train_data)))
                print("SGD: global mean is {}".format(global_mean))
                
            lr = self.sgd_learning_rate(0)
            for s in range(self.it):
                if round(self.it / 12) == self.it /12 and s % round(self.it / 12) == 0:
                    lr = self.sgd_learning_rate(s)
                    if verbose:
                        score = self.RMSE(self.train_data, U.dot(V.T) + biasU.reshape(-1,1) + biasV)
                        test_score = self.RMSE(self.test_data, U.dot(V.T) + biasU.reshape(-1,1) + biasV) if self.test_data is not None else -1
                        print("SGD: step {:8d}  ({:2d}% done). fit = {:.4f}, test_fit={:.4f}, learning_rate={:.5f}".format(
                            s+1, int(100 * (s+1)/self.it), score, test_score, lr))
                d,n,v = random.choice(self.train_data)
                d, n = d-1, n-1

                U_d = U[d,:]
                V_n = V[n,:]

                biasU_d = biasU[d]
                biasV_n = biasV[n]

                guess = U_d.dot(V_n) + biasU_d + biasV_n
            
                delta = v - guess

                try:
                    new_U_d = U_d + lr * (delta * V_n - reg*U_d)
                    new_V_n = V_n + lr * (delta * U_d - reg*V_n)

                    new_biasU_d = biasU_d + lr * ( delta - reg2*(biasU_d + biasV_n - global_mean))
                    new_biasV_n = biasV_n + lr * ( delta - reg2*(biasV_n + biasU_d - global_mean))
                    
                except FloatingPointError:
                    print("WARNING : FloatingPointError caught! Iteration skipped!")
                    continue
                else:
                    U[d,:] = new_U_d
                    V[n,:] = new_V_n
                    
                    biasU[d] = new_biasU_d
                    biasV[n] = new_biasV_n
            pickle.dump([U, V,  biasU, biasV], open(self.out_path, 'wb'))
        return U.dot(V.T) + biasU.reshape(-1,1) + biasV 

    def sgd_learning_rate(self, s):
        """
        Adjusting rate.
    
        s : iteration counter
        """
        rates = [0.035, 0.032, 0.029, 0.027, 0.012, 0.01, 0.0022, 0.002, 0.00055, 0.0005, 0.0001, 0.00002]
        progress = round((s / self.it) * 12)
        return rates[progress] * self.lrf
"""
sgd = SGD('../data/SGD13.pickle', 0.08, 0.04, 10, 60000000, 2.9, True)
sgd.train()
sgd.generatePredictions('../data/SGD13.csv')
"""

regs = [0.08]#[0.07, 0.075, 0.08, 0.085]
regs2 = [0.04]#[0.03, 0.035, 0.04, 0.045]
ks = [12]#[13, 17, 15, 14]#[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26] #[11, 12, 13, 14, 15, 16]
lrs = [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5]

for r1 in regs:
    for r2 in regs2:
        for k in ks:
            for lr in lrs:
                sgd = SGD('../data/var_k/SGD{0}_{1}_{2}_{3}.pickle'.format(r1, r2, k, lr), r1, r2, k, 60000000, lr, True)
                sgd.train()
                sgd.generatePredictions('../data/var_k/SGD{0}_{1}_{2}_{3}.csv'.format(r1, r2, k, lr))


#1 k=12, reg=0.083, reg2=0.04, lr=3.0 ===> 0.9269
#2 k=12, reg=0.08, reg2=0.04, lr=3.0 ===> 0.9240
#3 k=15, reg=0.08, reg2=0.04, lr=3.0 ===> 0.9157
#4 k=16, reg=0.08, reg2=0.04, lr=3.2 ===> 0.9132
#5 k=17, reg=0.08, reg2=0.04, lr=3.2 ===> 0.9109