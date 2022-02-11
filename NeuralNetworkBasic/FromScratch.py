import numpy as np

def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

def sigmoid(Z):
    A =  1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def initialize_params(dimensions):
    L = len(dimensions) #imagine this is 4
    parameters = {}
    for l in range(1,L): # for n in 1,2,3
        parameters['W'+str(l)] = np.random.randn(dimensions[l], dimensions[l-1]) /np.sqrt(dimensions[0]) #30,2000
        parameters['b'+str(l)] = np.zeros((dimensions[l], 1))
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b 
    cache = (A, W, b)
    return Z, cache

def activation_forward(A_prev, W, b, activation):
    
    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        
    cached = (linear_cache, activation_cache) # linear_cache = (A_prev, W, b), activation_cache = Z
    return A, cached #returns A, ((A_prev, W, b), Z)

def model_forward(X, params):
    parameters = params.copy()
    L = len(parameters)//2 #number of layers-1 = 3 in this case
    A = X
    caches = []
    for l in range(1,L): # for 1,2
        A_prev = A
        A, cache = activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation = 'relu')
        caches.append(cache) #Caches will be a two column vector with the first column containing (A_prev, W, b) and the second containing Z
    AL, cache = activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation = 'sigmoid')
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (1/m)*(-np.dot(Y,np.log(AL).T)-np.dot(1-Y,np.log(1-AL).T))
    cost = np.squeeze(cost)
    return cost

def sigmoid_backward(dA, cache):
    Z = cache
    s, cache = sigmoid(Z)
    dZ = dA*s*(1-s) #dZ (dL/dZ) is computed by running dL/dA * dA/dZ
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<0] = 0 #dZ (dL/dZ) is computed by running dL/dA*dA/dZ. Where Z > 0, derivative of relu is 1, where Z<0, d/dZ of relu is 0, so when the functions are multiplied, where Z is less than zero d 
    return dZ

def linear_backward(dZ, cache): 
    A_prev, W, b = cache
    m = A_prev.shape[1]
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
    L = len(caches) #caches have length 1 less than number of layers, example 4 layer model, L = 3
    Y = Y.reshape(AL.shape)
    grads = {}
    dAL = -np.divide(Y, AL)+np.divide(1-Y,1-AL)
    
    dA_prev, dW, db = activation_backward(dAL, caches[L-1], activation = 'sigmoid')# caches[3]
    grads['dA'+str(L-1)] = dA_prev #dA2
    grads['dW'+str(L)] = dW #dW3
    grads['db'+str(L)] = db #db3
    for l in reversed(range(L-1)): # in range 1,0
        dA_prev, dW, db = activation_backward(grads['dA'+str(l+1)], caches[l], activation = 'relu')
        grads['dA'+str(l)] = dA_prev #dA1
        grads['dW'+str(l+1)] = dW #dW2
        grads['db'+str(l+1)] = db #db2
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    params = parameters.copy()
    for l in range(L): # 0,1,2
        params['W'+str(l+1)] = params['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        params['b'+str(l+1)] = params['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]
    return params




        