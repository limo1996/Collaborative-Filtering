import os
import numpy as np 
from utils import loadRawData
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPRegressor

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
        print(f,w)
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

def shift2(files, rmses, out_file, target_rmse):
    """ Predicts ratings base on linear regression of previous ratings and their rmses. """
    assert len(files) == len(rmses)
    items, users, ratings = loadRawData(file=files[0])
    ratings2 = list([ratings])
    for f in files[1:]:
        _, _, r = loadRawData(file=f)
        ratings2.append(r)
    final_ratings = list()
    for i in range(0, len(ratings)):
        rat = list([ratings2[j][i] for j in range(0, len(files))])
         # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(rmses, rat)
        # Make predictions using the testing set
        final_ratings.append(regr.predict([[target_rmse]])[0])

    makePredictions(users, items, final_ratings, out_file)

def shift3(files, rmses, out_file, target_rmse):
    """ Predicts ratings with MLPRegressor trained on previous ratings and their rmses. """
    assert len(files) == len(rmses)
    items, users, ratings = loadRawData(file=files[0])
    ratings2 = list([ratings])
    for f in files[1:]:
        _, _, r = loadRawData(file=f)
        ratings2.append(r)
    final_ratings = list()
    for i in range(0, len(ratings)):
        rat = list([ratings2[j][i] for j in range(0, len(files))])
         # Create MLPRegressor object
        clf = MLPRegressor(alpha=0.01, hidden_layer_sizes = (5,5), max_iter = 1000, 
                 activation = 'relu', learning_rate = 'adaptive')
        # Train the model using the training sets
        clf.fit(rmses, rat)
        # Make predictions using the testing set
        final_ratings.append(clf.predict([[target_rmse]])[0])

    makePredictions(users, items, final_ratings, out_file)

#blend_files(['../data/SGD2.csv', '../data/svdpp.csv', '../data/baseline.csv'], [4, 1, 1], '../data/sgd_svdpp_bas.csv')             # ==> 0.98126
#blend_files(['../data/sgd_svdpp_bas.csv', '../data/svd_tun.csv'], [1, 1], '../data/sgd_svdpp_bas_svd.csv')                         # ==> 0.98842
#shift('../data/SGD2.csv', ['../data/svdpp.csv', '../data/baseline.csv', '../data/svd_tun.csv'], '../data/shift1_1.csv', 0.01)      # ==> 0.97943
#shift('../data/SGD2.csv', ['../data/svdpp.csv', '../data/baseline.csv', '../data/svd_tun.csv'], '../data/shift1_2.csv', 0.005)     # ==> 0.97945
#shift2(['../data/svd_tun.csv', '../data/SlopeOne.csv', '../data/baseline.csv', '../data/svdpp.csv', '../data/SGD2.csv'], 
#    [[1.00160], [0.99832], [0.99768], [0.99507], [0.97949]], '../data/shift2_1.csv', 0.9678)                                       # ==> 0.98465
#shift2(['../data/svd_tun.csv', '../data/SlopeOne.csv', '../data/baseline.csv', '../data/svdpp.csv', '../data/SGD2.csv'], 
#    [[1.00160], [0.99832], [0.99768], [0.99507], [0.97949]], '../data/shift2_2.csv', 0.975)                                        # ==> 0.98018
#shift2(['../data/svd_tun.csv', '../data/SlopeOne.csv', '../data/baseline.csv', '../data/svdpp.csv', '../data/SGD2.csv'], 
#    [[1.00160], [0.99832], [0.99768], [0.99507], [0.97949]], '../data/shift2_3.csv', 0.9777)                                       # ==> 0.97962
#shift2(['../data/svd_tun.csv', '../data/SlopeOne.csv', '../data/baseline.csv', '../data/svdpp.csv',  '../data/shift2_1.csv', 
#    '../data/sgd_svdpp_bas.csv', '../data/shift2_1.csv', '../data/shift2_2.csv', '../data/SGD2.csv', '../data/shift1_1.csv'], 
#    [[1.00160], [0.99832], [0.99768], [0.99507], [0.98465], [0.98126], [0.98018], [0.97962], [0.97948], [0.97945]], 
#    '../data/shift5.csv', 0.9777)                                                                                                  # ==> 0.98022
#shift3(['../data/svd_tun.csv', '../data/SlopeOne.csv', '../data/baseline.csv', '../data/svdpp.csv',  '../data/shift2_1.csv', 
# '../data/sgd_svdpp_bas.csv', '../data/shift2_1.csv', '../data/shift2_2.csv', '../data/SGD2.csv', '../data/shift1_1.csv'], 
#    [[1.00160], [0.99832], [0.99768], [0.99507], [0.98465], [0.98126], [0.98018], [0.97962], [0.97948], [0.97945]], 
#    '../data/shift7.csv', 0.9777)                                                                                                  # ==> 0.98022
#shift('../data/SGD2.csv', ['../data/svdpp.csv', '../data/baseline.csv', '../data/svd_tun.csv'], '../data/shift1_3.csv', 0.015)     # ==> 0.97943
#blend_files(['../data/SGD2.csv', '../data/SGD8.csv'], [5,4], '../data/SGD2_8.csv')                                                 # ==> 0.97817
#blend_files(['../data/SGD2.csv', '../data/SGD8.csv', '../data/SGD11.csv', '../data/SGD12.csv'], 
#    [8, 6, 7, 5], '../data/SGD2_8_11_12.csv')                                                                                      # ==> 0.97763
#blend_files(['../data/SGD2.csv', '../data/SGD8.csv', '../data/SGD11.csv', '../data/SGD12.csv'], 
#    [1, 1, 1, 1], '../data/SGD2_8_11_12_eq.csv')                                                                                   # ==> 0.97761
#blend_files(['../data/SGD2.csv', '../data/SGD8.csv', '../data/SGD11.csv', '../data/SGD12.csv', '../data/SGD2_8_11_12.csv', 
# '../data/SGD2_8_11_12_eq.csv'], [1, 1, 1, 1, 1, 1], '../data/SGD2_8_11_12_spec.csv')                                              # ==> 0.97761
#blend_files(['../data/SGD2.csv', '../data/SGD4.csv', '../data/SGD8.csv', '../data/SGD11.csv', '../data/SGD12.csv', 
# '../data/SGD13.csv'], [1, 1, 1, 1, 1, 1], '../data/SGD2_4_8_11_12_13.csv')                                                        # ==> 0.97739
#files = ['../data/var_k/{0}'.format(f) for f in os.listdir('../data/var_k')]
#blend_files(files, np.ones(len(files)), '../data/SGD_spec.csv')                                                                    # ==> 0.97723
#blend_files(files, np.ones(len(files)), '../data/SGD_spec3.csv')                                                                    # ==> 0.97718