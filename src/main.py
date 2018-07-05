import argparse
import numpy as np
import os

from blending import blend_files
from SGD import SGD
from plotting import plots

def bestSGD():
    r1, r2, k, lr = 0.04, 0.08, 12, 2.9
    sgd = SGD('bestSGD.pickle', r1, r2, k, 60000000, lr, True)
    sgd.train()
    sgd.generatePredictions('bestSGD.csv')

def bestBlending():
    files = ['../data/var_k/{0}'.format(f) for f in os.listdir('../data/var_k')]
    print('Blending', len(files), 'files from ../data/var_k directory.')
    blend_files(files, np.ones(len(files)), 'bestBlending.csv')

def gridSearch():
    """                  *** WARNING: Takes around 90 hours to run on Intel Core i5 Macbook Pro ***
    Grid search:
        - method to search for optimal parameters
        - runs all combinations of parameters below
        - outputs results into console
        - script in plotting/plots.py can parse output and plot nice figures :)
    """
    regs = [0.07, 0.075, 0.08, 0.085]
    regs2 = [0.03, 0.035, 0.04, 0.045]
    ks = [11, 12, 13, 14, 15, 16]
    lrs = [2.8, 2.9, 3, 3.1, 3.2]

    print('Generating grid search for:')
    print('k=', ks)
    print('reg=', regs)
    print('reg2=', regs2)
    print('lr=', lrs)
    for r1 in regs:
        for r2 in regs2:
            for k in ks:
                for lr in lrs:
                    sgd = SGD('../data/SGD{0}_{1}_{2}_{3}.pickle'.format(r1, r2, k, lr), r1, r2, k, 60000000, lr, False)
                    sgd.train()
                    #sgd.generatePredictions('../data/SGD{0}_{1}_{2}_{3}.csv'.format(r1, r2, k, lr))


def generatePlots():
    plots.plot4('../data/outputs/big_grid.txt')
    plots.plot_lr('../data/outputs/lr.txt')
    plots.plotk('../data/outputs/k.txt')

parser = argparse.ArgumentParser(description='Reproduction of results.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bestSGD', default=False, action='store_true',
    help='Reproduces submission file called bestSGD.csv with the best achieved RMSE for SGD.')
parser.add_argument('--bestBlending', default=False, action='store_true',
    help='Creates submission file named bestBlending.csv with the best achieved RMSE for Blending. Note that this script does not run SGD for all configurations but loads already produced predictions from data/var_k folder.')
parser.add_argument('--gridSearch', default=False, action='store_true',
    help='Runs grid search for the best SGD configuration.')
parser.add_argument('--generatePlots', default=False, action='store_true',
    help='Displays plots that you can see in the report.')
args = parser.parse_args()
assert sum([args.bestSGD, args.bestBlending, args.gridSearch, args.generatePlots]) == 1, 'more than one flag specified'

if args.bestSGD:
    bestSGD()
elif args.bestBlending:
    bestBlending()
elif args.gridSearch:
    gridSearch()
elif args.generatePlots:
    generatePlots()
