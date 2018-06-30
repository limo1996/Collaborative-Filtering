import numpy as np
from SGD import SGD

class SGD_SVD(SGD):
    """
        Not really working approach that runs SVD after SGD. Makes almost no improvement... 
    """
    def __init__(self, out_path, reg, reg2, k, it, lr, submission):
        SGD.__init__(self, out_path, reg, reg2, k, it, lr, submission)

    def run_svd(self, kk):
        """ Runs SVD on predictions matrix """
        self.kk = kk
        self.validate('Before SVD')
        # do SVD on them
        U, S, Vt = np.linalg.svd(self.predictions, full_matrices=True)
        print(self.predictions)
        self.predictions = self.reconstruct(U, S, Vt, kk, self.predictions.shape)
        print(self.predictions)
        self.validate('After SVD')

    def validate(self, msg):
        """ Prints current test and train RMSEs"""
        print(msg)
        score = self.RMSE(self.train_data, self.predictions)
        test_score = self.RMSE(self.test_data, self.predictions) if self.test_data is not None else -1
        print("SVD: fit = {:.4f}, test_fit={:.4f}, k={:.1f}".format(score, test_score, self.kk))

    def reconstruct(self, U, S, Vt, k, t_shape):
        """ reconstructs original matrix from factorization with k most significant features """
        S_ = np.zeros(t_shape)
        rows, cols = np.diag_indices(min(S_.shape))
        S_[rows, cols] = S
        return U[:,0:k].dot(S_[0:k,0:k].dot(Vt[0:k,:]))

sgd = SGD_SVD('../data/SGD_SVD_train.pickle', 0.08, 0.04, 12, 60000000, 2.9, False)
sgd.train()
sgd.run_svd(12)
#sgd.generatePredictions('../data/SGD_SVD_train.csv')