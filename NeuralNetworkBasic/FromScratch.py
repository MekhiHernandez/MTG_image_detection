import numpy as np

def relu(Z):
    A = np.max(0,Z)
    cache = Z
    return A, cache

def sigmoid(Z):
    A =  1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def initialize_params(dimensions):
    L = len(dimensions) #imagine this is 3
    parameters = {}
    for l in reversed(range(L)): # for n in 0, 1, 2
        parameters['W'+str(l+1)] = np.random.randn(dimensions[l+1], dimensions[l])/np.sqrt(dimensions[0]) 
        parameters['b'+str(l+1)] = np.zeros((dimensions[l], 1))
    return parameters 
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b