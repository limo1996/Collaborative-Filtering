import os
import re
import math
import numpy as np
import random
from six.moves import cPickle as pickle

# data matrix with fixed size
train_data = np.ndarray((10000,1000),dtype=float)
train_data.fill(float('nan'))
valid_data = {}

f = open("../data/data_train.csv", "rb")
content = f.readlines()

# 80% training data, 20% validation data
content = np.array(content[1:])
length = len(content)
idxs = random.sample(range(length), int(0.8*length))
train_content = content[idxs]
valid_content = np.delete(content, idxs)

# test for correct splitting
assert len(content) == len(train_content) + len(valid_content)
assert len(content) == len(set().union(train_content, valid_content))

# print info
print('{0} items selected for training'.format(len(train_content)))
print('{0} items selected for validation'.format(len(valid_content)))

def parse(line):
    """ parses line and returns parsed row, column and value """
    m = re.search('r(.+?)_c(.+?),(.+?)', line.decode('utf-8'))
    row = int(m.group(1)) - 1
    column = int(m.group(2)) - 1
    value = int(m.group(3))
    return row, column, value

# load each line from train data into matrix
for line in train_content:
    row, column, value = parse(line)
    train_data[row,column] = value

# load validation data into map since they are very sparse
for line in valid_content:
    row, column, value = parse(line)
    valid_data[(row, column)] = value
f.close()

# normalize data in each row -> normalize user's ratings
mean = np.nanmean(train_data, axis=1, keepdims=True)
train_data = train_data - mean

# fill in missing values with appropriate average
new_mean = np.nanmean(train_data, axis=1, keepdims=True)
inds = np.where(np.isnan(train_data))
train_data[inds] = np.take(new_mean, inds[0])
print(train_data)

# test that data are centered
new_mean = np.nanmean(train_data, axis=1)
assert all(m < 0.01 and m > -0.01 for m in new_mean)

# save matrix into pickle file for further usage
with open("../data/data.pickle", "wb") as f:
    save = { 'train_data' : train_data, 'valid_data' : valid_data, 'mean' : mean }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)