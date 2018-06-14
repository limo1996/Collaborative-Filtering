"""
This module descibes how to split a dataset into two parts A and B: A is for
tuning the algorithm parameters, and B is for having an unbiased estimation of
its performances. The tuning is done by Grid Search.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import random

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import GridSearchCV

from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

from utils import loadDataForSurprise, makePredictions
"""
# Load the full dataset.
data = loadDataForSurprise()
raw_ratings = data.raw_ratings

# shuffle ratings if you want
random.shuffle(raw_ratings)

# A = 90% of the data, B = 10% of the data
threshold = int(.9 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = A_raw_ratings  # data is now the set A

# Select your best algo with grid search.
print('Grid Search...')
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid_search.fit(data)

algo = grid_search.best_estimator['rmse']

# retrain on the whole set A
trainset = data.build_full_trainset()
algo.fit(trainset)

# Compute biased accuracy on A
predictions = algo.test(trainset.build_testset())
print('Biased accuracy on A,', end='   ')
accuracy.rmse(predictions)

# Compute unbiased accuracy on B
testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
predictions = algo.test(testset)
print('Unbiased accuracy on B,', end=' ')
accuracy.rmse(predictions)

"""
# ---------------------------------------
data = loadDataForSurprise()
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid_search.fit(data)
algo = grid_search.best_estimator['rmse']
# retrain on the whole set A
trainset = data.build_full_trainset()
algo.fit(trainset)
makePredictions(algo, '../data/svd_tun.csv')

# validate it
kf = KFold(random_state=0)
out = cross_validate(algo, data, ['rmse', 'mae'], kf)

# print errors
print("RMSE: {0}, MAE: {1}".format(np.mean(out['test_rmse']), np.mean(out['test_mae'])))