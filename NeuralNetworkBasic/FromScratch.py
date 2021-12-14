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
    cache = (A_prev, W, b)
    return (Z, cache)

def activation_forward(A_prev, W, b, activation):
    
    Z, linear_cache = linear_forward(A_prev, W, b) #cached just currently contains A_prev
    if activation == 'relu':
        A, activation_cache = relu(Z)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    cached = (linear_cache, activation_cache) # linear_cache = (A_prev, W, b), activation_cache = Z
    return A, cached #returns A, ((A_prev, W, b), Z)

def model_forward(X, params):
    parameters = params.copy()
    L = len(parameters)//2
    for l in reversed(range(L)):
        A, cached = activation_forward(X, parameters['W'+str(l)], parameters['b'+str(l)])


        