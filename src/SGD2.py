import os
import random
import numpy as np 
from six.moves import cPickle as pickle
from AlgoBase import AlgoBase

class SGD2(AlgoBase):
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
        split_ratio = 0.8 if not self.submission else 1
        self.split_data(0.8)
        predictions = self.sgd(True)
        predictions[predictions > 5.0] = 5.0
        predictions[predictions < 1.0] = 1.0
        self.predicitons = predictions

    def predict(self, user, item, verbose=False):
        """ Predicts rating for (user, item) pair """
        return self.predicitons[user, item]

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
                print("SGD2 solving started. Params: k={}, reg={}, reg2={}, lr={}".format(self.k, self.reg, self.reg2, self.lrf))
                print("SGD2: There are {} predictions given.".format(len(self.train_data[0])))
                print("SGD2: global mean is {}".format(global_mean))
                
            lr = self.sgd_learning_rate(0)
            for s in range(self.it):
                if round(self.it / 12) == self.it /12 and s % round(self.it / 12) == 0:
                    lr = self.sgd_learning_rate(s)
                    if verbose:
                        score = self.RMSE(self.train_data, U.dot(V.T) + biasU.reshape(-1,1) + biasV)
                        test_score = self.RMSE(self.test_data, U.dot(V.T) + biasU.reshape(-1,1) + biasV) if self.test_data is not None else -1
                        print("SGD2: step {:8d}  ({:2d}% done). fit = {:.4f}, test_fit={:.4f}, learning_rate={:.5f}".format(
                            s+1, int(100 * (s+1)/self.it), score, test_score, lr))
                for d,n,v in self.train_data:
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

sgd2 = SGD2('../data/SGD2_2.pickle', 0.08, 0.04, 17, 60, 2.8, False)
sgd2.train()
sgd2.predictions('../data/SGD2_2.csv')
#1 k=12, reg=0.083, reg2=0.04, lr=3.0 ===> 0.9269
#2 k=12, reg=0.08, reg2=0.04, lr=3.0 ===> 0.9258