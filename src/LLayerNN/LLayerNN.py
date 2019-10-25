import numpy as np

def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=300, print_cost=False):

    '''
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID

    Arguments:
    X -- data, numpy array of vectorized data with shape (n_x, m)
    Y -- "true" label vector of shape (1, m)
    layer_dims -- list containing the input size and each layer size
    learning_rate -- learning rate of the gradient descent update
    num_iterations -- number of iterations of the optimization loop
    print_cost -- print the cost every 100 steps

    Returns:
    parameters -- parameters learned by the model; can then be used to predict
    '''

    costs = []

    # Parameter initialization
    parameters = initialize_parameters_deep(layer_dims)

    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR->RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X,parameters)

        # Compute cost
        cost = compute_cost(AL,Y)

        # Backward propagation
        grads = L_model_backward(AL,Y,caches)

        # Update parameters
        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost and i%100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

    return parameters

def sigmoid(z):
    '''
    Inputs:
    z -- scalar or numpy array

    Return
    sigmoid(z), matches shape of input
    '''

    return 1/(1+np.exp(-z))

def relu(z):
    '''
    Rectified linear unit (ReLU)

    Inputs:
    z -- scalar or numpy array

    Return:
    np.max(0,z), matches shape of input
    '''

    return np.max(0,z)

def linear_forward(A,W,b):
    '''
    Implement the linear part of a layer's forward propagation

    Arguments:
    A -- activations from the previous layer (or input data)
    W -- weights matrix
    b -- bias vector

    Returns:
    Z -- linear result and input to the activation function
    cache -- python tuple containing "A", "W", and "b"
    '''

    Z = np.dot(W,A)+b

    cache = (A,W,b)

    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    '''
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from the previous layer (or input data) with shape 
        (size of previous layer, number of examples)
    W -- weights matrix with shape (size of current layer, size of previous layer)
    b -- bias vector with shape (size of current layer, 1)
    activation -- string indicating which activation to use: RELU or sigmoid

    Returns:
    A -- the output of hte activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
        stored for computing the backward pass efficiently
    '''

    if activation == 'sigmoid':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,linear_cache = sigmoid(Z)

    elif activation == 'relu':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)

    cache = (linear_cache,activation_cache)

    return A, cache

def initialize_parameters_deep(layer_dims):

    '''
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in the network

    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL"
    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    bl -- bias vector of shape (layer_dims[l],1)
    '''

    parameters = {}
    L = len(layer_dims)

    for l in range(1,L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

    return parameters

def L_model_forward(X,parameters):

    '''
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID

    Arguments:
    X -- data, numpy array of shape (input size (n_x), number of examples (m))
    parameters -- output of initialize_parameters_deep above()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing every cache of linear_activation_forward() (L-1 of them)
    '''

    caches = []
    A = X
    L = len(parameters) // 2

    # Implement [LINEAR->RELU]*(L-1). Add cache to caches list
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W' + str(l)],parameters['b' + str(l)],"relu")
        caches.append(cache)

    AL,cache = linear_activation_forward(A_prev,parameters['W' + str(L)],parameters['b' + str(L)],"sigmoid")
    caches.append(cache)

    return AL,caches


