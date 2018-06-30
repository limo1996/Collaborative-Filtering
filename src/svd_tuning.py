"""
This module descibes how to split a dataset into two parts A and B: A is for
tuning the algorithm parameters, and B is for having an unbiased estimation of
its performances. The tuning is done by Grid Search.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import random
import numpy as np
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import GridSearchCV

from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

from utils import loadDataForSurprise, makePredictions

# load the data
data = loadDataForSurprise()

# parameter intervals
param_grid = {'n_factors':[1, 2, 5, 20], 'n_epochs': [40, 50, 55, 60, 65], 'lr_all': [0.0075, 0.01, 0.4, 0.09, 0.11], 'reg_all': [0.0075, 0.01, 0.012, 0.015, 0.2]}

# do the grid search
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1)

# train 
grid_search.fit(data)

# best RMSE score
print(grid_search.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(grid_search.best_params['rmse'])

# save complete results as pandas dataframe
results_df = pd.DataFrame.from_dict(grid_search.cv_results)
results_df.to_csv('../data/svd_tun_df2.csv', sep='\t', encoding='utf-8')

# get algorithm instance with best parameters
algo = grid_search.best_estimator['rmse']

# retrain on the whole set A
trainset = data.build_full_trainset()

# train the algorithm
algo.fit(trainset)

# and save the predictions
makePredictions(algo, '../data/svd_tun2.csv')

# validate it
kf = KFold(random_state=0)
out = cross_validate(algo, data, ['rmse', 'mae'], kf)

# print errors
print("RMSE: {0}, MAE: {1}".format(np.mean(out['test_rmse']), np.mean(out['test_mae'])))

# 1.00195478756843
# {'n_factors': 1, 'n_epochs': 60, 'lr_all': 0.01, 'reg_all': 0.01}
# RMSE: 1.0020650570969798, MAE: 0.8081256801637089