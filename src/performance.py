import os
import numpy as np
from six.moves import cPickle as pickle
from sklearn.metrics import mean_squared_error
from math import sqrt

# evaluates performace with RMSE 
def performance(prediction_matrix, model_name):
    pfile = 'performance.txt'
    # create file if does not exist
    if not os.path.exists(pfile):
        f = open(pfile, 'w+')
        f.close()

    # load the expected data
    with open('../data/data.pickle', 'rb') as f:
        data = pickle.load(f)
        test_data = data['valid_data']

    # create two lists with expected and actual values
    expected = []
    actual = []
    for key, value in test_data.items():
        expected.append(value)
        actual.append(prediction_matrix[key[0],key[1]])
    
    # compute rmse, print it, append it to the file and return it
    rmse = sqrt(mean_squared_error(expected, actual))
    print('RMSE: {0} of {1}'.format(rmse, model_name))
    with open(pfile, 'a') as f:
        f.write('{0}\t{1}\n'.format(rmse, model_name))
    return rmse