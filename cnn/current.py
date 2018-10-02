import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklean.utils import shuffle


def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(p, t):
    return np.mean(p!=t)

def flatten(x):
    N = x.shape[-1]
    flat = np.zeros((N, 3072))
    for i in range(N):
        flat[i] = x[:,:,:,i].reshape(3072)
    return flat

def main():
    train = loadmat('./deep_learning/train_32x32.mat')
    test = loadmat('./deep_learning/test_32x32.mat')

    Xtrain = flatten(train['x'].astype(np.float32)/255)
    Ytrain = train['y'].flatten()-1
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    Xtest = flatten(test['x'].astype(np.float32) / 255)
    Ytest = test['y'].flatten() - 1
    Ytest_ind = y2indicator(Ytest)

    max_iter = 20
    print_period = 10
    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N / batch_sz

    M1 = 1000
    M2 = 500
    K = 10


#df['x1x2'] = df.apply(labmdarow: row['x1'] * row['x2'], axis = 1)


