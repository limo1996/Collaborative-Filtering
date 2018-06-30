"""
This module descibes how to split a dataset into two parts A and B: A is for
tuning the algorithm parameters, and B is for having an unbiased estimation of
its performances. The tuning is done by Grid Search.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import random
import numpy as np

from surprise import SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import GridSearchCV

from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

from utils import loadDataForSurprise, makePredictions

data = loadDataForSurprise()

param_grid = {'n_factors':[100,200], 'n_epochs': [15, 25], 'lr_all': [0.002, 0.005]}

grid_search = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=3)

grid_search.fit(data)

algo = grid_search.best_estimator['rmse']

# retrain on the whole set A
trainset = data.build_full_trainset()

algo.fit(trainset)

makePredictions(algo, '../data/svdpp_tun.csv')

# validate it
kf = KFold(random_state=0)
out = cross_validate(algo, data, ['rmse', 'mae'], kf)

# print errors
print("RMSE: {0}, MAE: {1}".format(np.mean(out['test_rmse']), np.mean(out['test_mae'])))