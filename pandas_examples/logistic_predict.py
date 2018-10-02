import numpy as np
from process import get_binary_data

X, Y = get_binary_data()

D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1/(1 + np.exp(-a))

def forward(x,w,b):
    return sigmoid(x.dot(w) + b)

P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

def classification_rate(Y, P):
    return np.mean(Y == P)

print ("Score: ", classification_rate(Y, predictions))


