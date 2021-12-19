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
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    cached = (linear_cache, activation_cache) # linear_cache = (A_prev, W, b), activation_cache = Z
    return A, cached #returns A, ((A_prev, W, b), Z)

def model_forward(X, params):
    parameters = params.copy()
    L = len(parameters)//2
    A_prev = X
    caches = []
    for l in reversed(range(L-1)):
        A_prev, cache = activation_forward(A_prev, parameters['W'+str(l+1)], parameters['b'+str(l+1)])
        caches.append(cache) #Caches will be a two column vector with the first column containing (A_prev, W, b) and the second containing Z
    AL, cache = activation_forward(A_prev, parameters['W'+str(L)], parameters['b'+str(L)])
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(1/m)*(np.sum(Y*np.log(AL))+np.sum((1-Y)*np.log(1-AL)))
    return cost

def sigmoid_backward(dA, cache):
    Z = cache
    s = sigmoid(Z)
    dZ = dA*s*(1-s) #dZ (dL/dZ) is computed by running dL/dA * dA/dZ
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<0] = 0 #dZ (dL/dZ) is computed by running dL/dA*dA/dZ. Where Z > 0, derivative of relu is 1, where Z<0, d/dZ of relu is 0, so when the functions are multiplied, where Z is less than zero d 
    return dZ

def linear_backward(dZ, cache): 
    A_prev, W, b = cache
    m = A_prev.shape(1)
    dA_prev = (1/m)*np.dot(W.T, dZ) 
    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
    return dA_prev, dW, db

def activation_backward(dA, caches, activation):
    linear_cache, activation_cache = caches
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db 

def model_backward(AL, Y, caches): 
    L = len(caches) #caches have length 1 less than number of layers, example 5 layer model, L = 4
    Y = Y.reshape(AL.shape)
    grads = {}
    dAL = -np.divide(Y, AL)+np.divide(1-Y,1-AL)
    dA_prev, dW, db = activation_backward(dAL, caches[L-1], activation = 'sigmoid')# caches[3]
    grads['dA'+str(L-1)] = dA_prev #dA3
    grads['dW'+str(L)] = dW #dW4
    grads['db'+str(L)] = db #db4
    for l in reversed(range(L-1)): # in range 2,1,0
        dA_prev, dW, db = activation_backward(grads['dA'+str(l+1)], caches[l], activation = 'relu')
        grads['dA'+str(l)] = dA_prev #dA2
        grads['dW'+str(l+1)] = dW #dW3
        grads['db'+str(l+1)] = db #db3
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    params = parameters.copy()
    for l in range(L): # 0,1,2
        params['W'+str(l+1)] = params['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        params['b'+str(l+1)] = params['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]
    return params




        