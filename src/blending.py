import numpy as np 
from utils import loadRawData

def blend_files(files, weights, out_file):
    """ 
        Combines results from files with weights passed as second argument.
        Resulting ratings are written to out_file.
     """
    assert len(files) == len(weights)
    sw = np.sum(weights)
    items, users, ratings = loadRawData(file=files[0])
    ratings = weights[0] * np.array(ratings)
    for f,w in zip(files[1:], weights[1:]):
        _, _, r = loadRawData(file=f)
        ratings = ratings + w * np.array(r)
    ratings = ratings / sw
    makePredictions(users, items, ratings, out_file)

def makePredictions(users, items, ratings, out_file):
    """ Writes arguments to out_file in submission format. """
    with open(out_file, 'w+') as f:
        f.write('Id,Prediction\n')
        for u,i,r in zip(users, items, ratings):
            f.write('r{0}_c{1},{2}\n'.format(u,i,r))

def allLower(data, i, value):
    """ Returns True if all values in i-th row are smaller than value """
    all_l = True
    for j in range(0, len(data)):
        all_l = all_l and data[j][i] < value
    return all_l

def allGreater(data, i, value):
    """ Returns True if all values in i-th row are greater than value """
    all_g = True
    for j in range(0, len(data)):
        all_g = all_g and data[j][i] > value
    return all_g

def shift(target, files, out_file, factor):
    """ Shifts rating of target by factor to left/right if all ratings in files are smaller/greater than it """
    items, users, ratings = loadRawData(file=target)
    ratings = np.array(ratings)
    ratings2 = list()
    for f in files:
        _, _, r = loadRawData(file=f)
        ratings2.append(r)
    for i in range(0, len(ratings)):
        if allLower(ratings2, i, ratings[i]):
            ratings[i] = ratings[i] + factor
        elif allGreater(ratings2, i, ratings[i]):
            ratings[i] = ratings[i] - factor
    makePredictions(users, items, ratings, out_file)

#blend_files(['../data/SGD2.csv', '../data/svdpp.csv', '../data/baseline.csv'], [4, 1, 1], '../data/sgd_svdpp_bas.csv') # ==> 0.98126
#blend_files(['../data/sgd_svdpp_bas.csv', '../data/svd_tun.csv'], [1, 1], '../data/sgd_svdpp_bas_svd.csv') # ==> 0.98842
#shift('../data/SGD2.csv', ['../data/svdpp.csv', '../data/baseline.csv', '../data/svd_tun.csv'], '../data/shift.csv', 0.01) # ==> 0.97943