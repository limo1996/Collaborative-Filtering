import re
import itertools
import numpy as np
import matplotlib.pyplot as plt 

FILE = '../../data/Output4.txt'

with open(FILE, 'rb') as f:
    content = f.readlines()

relevant_content = []

for line in content:
    line = line.decode('utf-8')
    if '91% done' in line or line.startswith('SGD solving started. Params:'):
        relevant_content.append(line)

def parse_lines(c, i):
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
    results = {}
    ks = []
    lrs = []
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
    plt.show()

# returns sorted set
def sortedSet(x):
    return list(sorted(set(x)))

def plot4(rc):
    results = {}
    ks = []
    regs = []
    regs2 = []
    lrs = []
    for i in range(0,len(rc),2):
        k, reg, reg2, lr, test_fit, _ = parse_lines(rc, i)
        ks.append(k)
        regs.append(reg)
        regs2.append(reg2)
        lrs.append(lr)
        results[(k,reg,reg2,lr)] = test_fit
    ks, regs, regs2, lrs = sortedSet(ks), sortedSet(regs), sortedSet(regs2), sortedSet(lrs)
    print(ks, regs, regs2, lrs)
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
    print(xt, yt)
    locs = np.arange(0, y_len) + 0.5
    plt.xticks(locs, xt)
    locs = np.arange(0, x_len) + 0.5
    print(locs)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=6)
    plt.yticks(locs, yt)
    plt.ylabel(r'$(k,\delta)$')
    plt.xlabel(r'$(\Lambda_1,\Lambda_2)$')
    plt.show()

def stats(rc):
    stat = []
    for i in range(0,len(rc),2):
        k, reg, reg2, lr, test_fit, fit = parse_lines(rc, i)
        stat.append((k, reg, reg2, lr, test_fit, fit))
    stat = sorted(stat, key=lambda x: x[4])
    for st in stat:
        print('(k={0}, reg={1}, reg={2}, lr={3}) ==> ({4}, {5})'.format(st[0], st[1], st[2], st[3], st[5], st[4]))

stats(relevant_content)
#plot4(relevant_content)