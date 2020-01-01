import numpy as np
import os

class LLayerNN:

    def __init__(self,data_path=None,learning_rate=None,num_iterations=None,verbose=None):

        if data_path is None:
            print("Please provide a path to data!")
            return 1
        else:
            self.data_path = data_path
            self.read_data_path()

        if verbose is None:
            self.verbose = False

        if learning_rate is None:
            self.learning_rate = 0.0075

        if num_iterations is None:
            self.num_iterations = 300

        if self.verbose == True:
            print("Reading data from: " + self.data_path.split("/")[-2:-1])
            print("Learning rate: " + str(self.learning_rate))
            print("Number of iterations: " + str(self.num_iterations))

    def read_data_path(self):

        self.file_list = os.listdir(self.data_path)
        
        for fname in self.file_list:

            if fname.split(".")[-1] == "train":

                

            elif fname.split(".")[-1] == "test":



            elif fname.split(".")[-1] == "dev":



            else:

                continue

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

    def compute_cost(AL, Y):

        '''
        Implement the cost function

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector, shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        '''

        m = Y.shape[1]

        # Complute loss from AL and Y
        cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m

        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost

    def linear_backward(dZ, cache):

        '''
        Implement the linear portion of backward propagation for a single layer (layer l)
        
        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        '''

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ,A_prev.T)/m
        db = np.sum(dZ,axis=1,keepdims=True)/m
        dA_prev=np.dot(W.T,dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev,dW, db

    def linear_activation_backward(dA, cache, activation):

        '''
        Implement the backward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        dA -- post-activation gradient for the current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previouslayer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with reqpect to b (current layer l), same shape as b
        '''

        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu_backward(dA,activation_cache)
            dA_prev, dW, db = linear_backward(dZ,linear_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA,activation_cache)
            dA_prev, dW, db = linear_backward(dZ,linear_cache)

        return dA_prev,dW, db

    def L_model_backward(AL, Y, caches):

        '''
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of hte forward propagation (L_model_forward())
        Y -- true "label" vector
        caches -- list of caches containing:
                Every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1))
                The cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

        Returns:
        grads -- a dictionary with the gradients
                grads["dA" + str(l)] 
                grads["dW" + str(l)]
                grads["db" + str(l)]
        '''

        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y,AL) - np.divide(1-Y,1-AL))

        current_cache = caches[L-1]

        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):

            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l+1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp

        return grads

    def update_parameters(parameters, grads, learning_rate):

        '''
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, outputs of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                parameters["W" + str(l)]
                parameters["b" + str(l)]
        '''

        L = len(parameters)
        
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

        return parameters
