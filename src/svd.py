import os
import re
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

# try all k = 7 seems to be the best
k = 7
# reconstruct matrix
predicted = reconstruct(U, S, Vt, k, train_data.shape) + mean
# evaluate offline performance
performance(predicted, 'SVD baseline k = {0}'.format(k))  

def parse(line):
    """ parses line and returns parsed row, column and value """
    m = re.search('r(.+?)_c(.+?),(.+?)', line.decode('utf-8'))
    row = int(m.group(1))
    column = int(m.group(2))
    value = int(m.group(3))
    return row, column, value

with open('../data/svd.csv', 'w+') as f:
    f.write('Id,Prediction\n')
    with open('../data/sampleSubmission.csv', 'rb') as f2:
        content = f2.readlines()
        content = content[1:]
        for line in content:
            if line:
                row, column, value = parse(line)
                f.write('r{0}_c{1},{2}\n'.format(row, column, predicted[row-1,column-1]))