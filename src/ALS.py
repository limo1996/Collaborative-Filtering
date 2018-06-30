import os
import random
import numpy as np
from six.moves import cPickle as pickle
from AlgoBase import AlgoBase


class ALS(AlgoBase):
    """
    Collaborative filtering algorithm for prediction of user item preferences.
    Parameters:
        out_path: Path for saving U, V computed matrices.
        k : the number of features to use
        it: number of iterations
        lr: learning rate factor
        submission: if true than all data are train data otherwise 80%
    """

    def __init__(self, out_path, k, it, lr, submission):
        self.out_path = out_path
        self.k = k
        self.it = it
        self.lrf = lr
        self.submission = submission
        AlgoBase.__init__(self)

    def train(self):
        split_ratio = 0.8 if not self.submission else 1
        self.split_data(0.8)
        predictions = self.als(True)
        predictions[predictions > 5.0] = 5.0
        predictions[predictions < 1.0] = 1.0
        self.predictions = predictions

    def predict(self, user, item, verbose=False):
        """ Predicts rating for (user, item) pair """
        return self.predictions[user, item]

    def als(self, verbose):
        """
        Alternating Least Squares (ALS) predictor.

        Training set is list of tuples where each tuple contains item id, user id and rating

        k : the number of features to use
        lr : learning rate

        verbose = if true process is reported

        """
        if verbose:
            print("ALS solving started. Params: k={}, lr={}".format(self.k, self.lrf))
            print("ALS: There are {} predictions given.".format(len(self.train_data[0])))
        users, items = 10000, 1000

        X = np.ndarray((users, items))
        X.fill(float('nan'))
        for user, item, val in self.train_data:
            X[user - 1][item - 1] = val
        mean = np.nanmean(X, axis=1, keepdims=True)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(mean, inds[0])
        m, n = X.shape
        U = 5 * np.random.rand(m, self.k)
        V = 5 * np.random.rand(self.k, n)
        for i in range(self.it):
            A = np.dot(V, V.T) + self.lrf * np.eye(self.k)
            B = np.dot(V, X.T)
            U = np.linalg.solve(A,B).T

            C = np.dot(U.T, U) + self.lrf * np.eye(self.k)
            D = np.dot(U.T, X)
            V = np.linalg.solve(C, D)
            if verbose and i % 50 == 0:
                score = self.RMSE(self.train_data, np.dot(U,V))
                test_score = self.RMSE(self.test_data, np.dot(U,V)) if self.test_data is not None else -1
                print("{}th iteration is completed fit = {:.4f}, test_fit={:.4f}".format(i, score, test_score))

        X_h = np.dot(U, V)
        X_h *= float(5) / np.max(X_h)

        return X_h



als = ALS(r"C:\Users\Raja\Desktop\CIL\project\ALS.pickle", 20, 1000, 0.5, False)
als.train()
als.genPredictions(r"C:\Users\Raja\Desktop\CIL\project\als_predictions.csv")