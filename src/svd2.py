import numpy as np
from surprise import SVD

from surprise.model_selection import KFold
from surprise.model_selection import cross_validate

from utils import makePredictions, loadDataForSurprise

# load the train data in surprise format
data = loadDataForSurprise()

# retrieve the trainset.
trainset = data.build_full_trainset()

# create SVD algorithm and train it
algo = SVD(n_factors=14, n_epochs=50, lr_all=0.09, reg_all=0.084)
algo.fit(trainset)

# save the predictions
makePredictions(algo, '../data/svd3.csv')

# validate it
kf = KFold(random_state=0)
out = cross_validate(algo, data, ['rmse', 'mae'], kf)

# print errors
print("RMSE: {0}, MAE: {1}".format(np.mean(out['test_rmse']), np.mean(out['test_mae'])))

# factors=1, epochs=50 ==> RMSE=0.9985101082212786