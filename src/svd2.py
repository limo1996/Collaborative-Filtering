import os
import re
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from six.moves import cPickle as pickle
from performance import performance

from tabulate import tabulate
import pandas as pd
from surprise import SVD
from surprise import SVDpp
from surprise import SlopeOne
from surprise import Dataset
from surprise import Reader

def parse(line):
    """ parses line and returns parsed row, column and value """
    m = re.search('r(.+?)_c(.+?),(.+?)', line.decode('utf-8'))
    row = int(m.group(1))
    column = int(m.group(2))
    value = int(m.group(3))
    return row, column, value

itemID = []
userID = []
rating = []

with open('../data/data_train.csv', 'rb') as f:
    content = f.readlines()
    content = content[1:]
    for line in content:
        if line:
            row, column, value = parse(line)
            itemID.append(column)
            userID.append(row)
            rating.append(value)

# Creation of the dataframe. Column names are irrelevant.
ratings_dict = {'itemID': itemID,
                'userID': userID,
                'rating': rating}
df = pd.DataFrame(ratings_dict)

# The columns must correspond to user id, item id and ratings (in that order).
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader=reader)

# Retrieve the trainset.
trainset = data.build_full_trainset()

algo = SVDpp()
algo.fit(trainset)

with open('../data/svd2.csv', 'w+') as f:
    f.write('Id,Prediction\n')
    with open('../data/sampleSubmission.csv', 'rb') as f2:
        content = f2.readlines()
        content = content[1:]
        for line in content:
            if line:
                row, column, value = parse(line)
                uid = row
                iid = column
                pred = algo.predict(uid, iid, verbose=False)
                f.write('r{0}_c{1},{2}\n'.format(row, column, pred.est))