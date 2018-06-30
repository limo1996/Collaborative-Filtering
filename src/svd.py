import os
import re
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import buildMatrix
from AlgoBase import AlgoBase

class SVD(AlgoBase):
    """
    Naive SVD for collaborative filtering implementation.

    k: top k dimensions to keep.
    """
    def __init__(self, k):
        self.k = k
        AlgoBase.__init__(self)

    def train(self):
        self.split_data(0.8)
        matrix = buildMatrix(self.train_data)
        # do SVD on them
        U, S, Vt = np.linalg.svd(matrix, full_matrices=True)
        self.eigvals = S**2 / np.cumsum(S)[-1]
        self.sing_vals = np.arange(matrix.shape[1]) + 1
        self.predictions = self.reconstruct(U, S, Vt, self.k, matrix.shape)

    def predict(self, user, item, verbose=False):
        """ Predicts rating for (user, item) pair """
        return self.predictions[user, item]

    def plot_scree(self):
        """ Plots scree plot """
        # Display scree plot
        fig = plt.figure(figsize=(8,5))
        plt.plot(self.sing_vals, self.eigvals, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        # I don't like the default legend so I typically make mine like below, e.g.
        # with smaller fonts and a bit transparent so I do not cover up data, and make
        # it moveable by the viewer in case upper-right is a bad place for it 
        leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)
        plt.show()

    def reconstruct(self, U, S, Vt, k, t_shape):
        """ reconstructs original matrix from factorization with k most significant features """
        S_ = np.zeros(t_shape)
        rows, cols = np.diag_indices(min(S_.shape))
        S_[rows, cols] = S
        return U[:,0:k].dot(S_[0:k,0:k].dot(Vt[0:k,:]))

    def evaluate(self):
        """ Evaluates accuracy of predictions """
        print('RMSE: ', self.RMSE(self.test_data, self.predictions))

svd = SVD(1)
svd.train()
svd.plot_scree()
svd.evaluate()