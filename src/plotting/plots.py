import re
import itertools
import numpy as np
import matplotlib.pyplot as plt 

# file to load content for
FILE = '../../data/Output3.txt'

with open(FILE, 'rb') as f:
    content = f.readlines()

relevant_content = []

for line in content:
    line = line.decode('utf-8')
    if '91% done' in line or line.startswith('SGD solving started. Params:'):
        relevant_content.append(line)

def parse_lines(c, i):
    """ 
        Parses two consecutive relevant lines and returns parsed tuple of (k, reg, reg2, lr, test_fit, fit)  
    """

    m = re.search('SGD solving started. Params: k=(.+?), reg=(.+?), reg2=(.+?), lr=(.+?)\n', c[i])
    k = int(m.group(1))
    reg = float(m.group(2))
    reg2 = float(m.group(3))
    lr = float(m.group(4))
    idx = c[i+1].find('test_fit=') + len('test_fit=')
    test_fit = float(c[i+1][idx:idx+6])
    fit = float(c[i+1][idx-17:idx-11])
    return k, reg, reg2, lr, test_fit, fit

def plot2(rc):
    """ Plots meshgrid plot of ks and lrs with fixed reg and reg2 """

    results, ks, lrs = {}, [], []
    for i in range(0,len(rc),2):
        k, _, _, lr, test_fit, _ = parse_lines(rc, i)
        results[(k,lr)] = test_fit
        ks.append(k)
        lrs.append(lr)

    x, y = np.meshgrid(ks,lrs)
    c = np.ndarray(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            c[i][j] = results[(x[i][j], y[i][j])]

    plt.pcolormesh(x, y, c, cmap='RdBu')
    plt.colorbar()
    plt.savefig('meshgrid2.pdf', format='pdf')
    plt.show()

# returns sorted set
def sortedSet(x):
    return list(sorted(set(x)))

def plot4(rc):
    """ Prints meshgrid plot from grid search of 4 parameters = k, reg, reg2, lr """
    results, ks, regs, regs2, lrs = {}, [], [], [], []
    for i in range(0,len(rc),2):
        k, reg, reg2, lr, test_fit, _ = parse_lines(rc, i)
        ks.append(k)
        regs.append(reg)
        regs2.append(reg2)
        lrs.append(lr)
        results[(k,reg,reg2,lr)] = test_fit
    ks, regs, regs2, lrs = sortedSet(ks), sortedSet(regs), sortedSet(regs2), sortedSet(lrs)
    #print(ks, regs, regs2, lrs)
    x_len = len(ks) * len(lrs)
    y_len = len(regs) * len(regs2)
    C = np.ndarray((x_len, y_len))
    i, j = 0, 0
    for k, lr in itertools.product(ks, lrs):
        j=0
        for reg, reg2 in itertools.product(regs, regs2):
            C[i][j]=results[(k,reg,reg2,lr)]
            j=j+1
        i=i+1
    X, Y = np.meshgrid(np.arange(y_len + 1), np.arange(x_len + 1))
    print(X.shape, Y.shape, C.shape)
    plt.pcolormesh(X, Y, C, cmap='YlOrRd_r')
    plt.colorbar()
    xt = ['({0},{1})'.format(x,y) for x, y in itertools.product(regs, regs2)]
    yt = ['({0},{1})'.format(x,y) for x, y in itertools.product(ks, lrs)]
    #print(xt, yt)
    locs = np.arange(0, y_len) + 0.5
    plt.xticks(locs, xt)
    locs = np.arange(0, x_len) + 0.5
    #print(locs)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=6)
    plt.yticks(locs, yt)
    plt.ylabel(r'$(k,\delta)$')
    plt.xlabel(r'$(\Lambda_1,\Lambda_2)$')
    plt.savefig('meshgrid4.pdf', format='pdf')
    plt.show()

def plotk(rc):
    """ Prints plot of k with all other parameters fixed """
    _, reg, reg2, lr, _, _ = parse_lines(rc, 0)
    ks, trains, tests = [], [], []
    for i in range(0, len(rc), 2):
        k, _, _, _, test_fit, train_fit, = parse_lines(rc, i)
        ks.append(k)
        trains.append(train_fit)
        tests.append(test_fit)

    ks, trains, tests = zip(*sorted(zip(ks, trains, tests), key=lambda x: x[0]))
    line1 = plt.plot(ks, trains, 'c')
    line2 = plt.plot(ks, tests, 'm')
    plt.legend(['Train RMSE', 'Test RMSE'])
    plt.grid(True, axis='y')
    ys = np.arange(round(np.min(trains), 2) - 0.01, round(np.max(tests), 2) + 0.01, 0.01)
    print(ys)
    plt.yticks(ys)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.savefig('plotk.pdf', format='pdf')
    plt.show()

def plot_lr(rc):
    """ Prints plot of lr with all other parameters fixed """
    k, reg, reg2, _, _, _ = parse_lines(rc, 0)
    lrs, trains, tests = [], [], []
    for i in range(0, len(rc), 2):
        _, _, _, lr, test_fit, train_fit, = parse_lines(rc, i)
        lrs.append(lr)
        trains.append(train_fit)
        tests.append(test_fit)

    lrs, trains, tests = zip(*sorted(zip(lrs, trains, tests), key=lambda x: x[0]))
    line1 = plt.plot(lrs, trains, 'y')
    line2 = plt.plot(lrs, tests, 'k')
    plt.legend(['Train RMSE', 'Test RMSE'])
    plt.grid(True, axis='y')
    ys = np.arange(round(np.min(trains), 2) - 0.01, round(np.max(tests), 2) + 0.01, 0.01)
    print(ys)
    plt.yticks(ys)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel(r'$\delta$')
    plt.ylabel('RMSE')
    plt.savefig('plot_lr.pdf', format='pdf')
    plt.show()

def stats(rc):
    """ Prints all collected results sorted from best test fit to the worst. """
    stat = []
    for i in range(0,len(rc),2):
        k, reg, reg2, lr, test_fit, fit = parse_lines(rc, i)
        stat.append((k, reg, reg2, lr, test_fit, fit))
    stat = sorted(stat, key=lambda x: x[4])
    for st in stat:
        print('(k={0}, reg={1}, reg={2}, lr={3}) ==> ({4}, {5})'.format(st[0], st[1], st[2], st[3], st[5], st[4]))

stats(relevant_content)
#plot4(relevant_content)
#plotk(relevant_content)
#plot_lr(relevant_content)
#plot4(relevant_content)
plot2(relevant_content)