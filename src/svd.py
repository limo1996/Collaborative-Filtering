import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from six.moves import cPickle as pickle
from performance import performance

# load data from the file
with open('../data/data.pickle', 'rb') as f:
    data = pickle.load(f)

train_data = data['train_data']
valid_data = data['valid_data']
mean = data['mean']

# do SVD on them
U, S, Vt = np.linalg.svd(train_data, full_matrices=True)
eigvals = S**2 / np.cumsum(S)[-1]

"""
# Display scree plot
fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(train_data.shape[1]) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
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
"""

# reconstructs original matrix from factorization with k most significant features
def reconstruct(U, S, Vt, k, t_shape):
    S_ = np.zeros(t_shape)
    rows, cols = np.diag_indices(min(S_.shape))
    S_[rows, cols] = S

    return U[:,0:k].dot(S_[0:k,0:k].dot(Vt[0:k,:]))

# try all k's around "knee"
for k in range(2, 15):
    predicted = reconstruct(U, S, Vt, k, train_data.shape) + mean
    performance(predicted, 'SVD baseline k = {0}'.format(k))  