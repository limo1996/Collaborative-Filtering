import random
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from utils import loadRawData, makePredictions

class AlgoBase(object):
    """
    Base class for all Collaborative filtering algorithms.

    Takes care about data loading and submission making.
    """

    def __init__ (self):
        """
        Loads data from hard defined path in format: 

            self.data : list of tuple(userID: int, itemID: int, rating: int)

        where all three lists are of the same length and i-th item represents
        item index, user index and rating of the item made by the user
        """

        data = loadRawData()
        self.global_mean = np.mean(data[2])
        self.data = list(zip(data[1], data[0], data[2]))
    
    def predictions(self, file):
        """ 
        Makes predictions according to the sampleSubmissions.csv file.
        """
        makePredictions(self, file, est=False)

    def split_data(self, ratio):
        """
        Splits data into training and testing samples by ratio.

        - ratio: between [0,1] specifying percentage of data that will be used for training.
                 Other part will be used for testing.
        """
        items = round(len(self.data) * ratio)
        random.shuffle(self.data)
        self.train_data = self.data[:items]
        self.test_data = self.data[items:]

    def RMSE(self, test_data, predictions):
        """
        Computes root mean square error on test data.
        """
        # create two lists with expected and actual values
        expected = []
        actual = []
        for d,n,v in test_data:
            expected.append(v)
            actual.append(predictions[d-1,n-1])
    
        # compute rmse and return it
        return sqrt(mean_squared_error(expected, actual))