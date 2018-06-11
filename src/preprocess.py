import os
import re
import math
import numpy as np
from six.moves import cPickle as pickle

# data matrix with fixed size
data = np.ndarray((10000,1000),dtype=float)
data.fill(float('nan'))
f = open("../data/data_train.csv", "rb")
content = f.readlines()

# load each line into matrix
for line in content[1:]:
    m = re.search('r(.+?)_c(.+?),(.+?)', line.decode('utf-8'))
    row = int(m.group(1)) - 1
    column = int(m.group(2)) - 1
    value = int(m.group(3))
    data[row,column] = value
f.close()

# normalize data in each row -> normalize user's ratings
mean = np.nanmean(data, axis=1, keepdims=True)
data = data - mean

# fill in missing values with appropriate average
mean = np.nanmean(data, axis=1, keepdims=True)
inds = np.where(np.isnan(data))
data[inds] = np.take(mean, inds[0])
print(data)

# test that data are centered
mean = np.nanmean(data, axis=1)
assert all(m < 0.01 and m > -0.01 for m in mean)

# save matrix into pickle file for further usage
with open("../data/data.pickle", "wb") as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)