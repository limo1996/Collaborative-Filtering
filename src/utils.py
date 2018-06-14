import os
import re

import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import Reader

def parse(line):
    """ parses line and returns parsed row, column and value """
    m = re.search('r(.+?)_c(.+?),(.+?)', line.decode('utf-8'))
    row = int(m.group(1))
    column = int(m.group(2))
    value = int(m.group(3))
    return row, column, value

def loadDataForSurprise():
    """ Loads and returns data in surprise format """
    itemID = []
    userID = []
    rating = []

    # parse data file into three arrays
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
    return data


def makePredictions(algo, file):
    """ Makes predictions file according to sampleSubmission file from algorithm provided """
    with open(file, 'w+') as f:
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