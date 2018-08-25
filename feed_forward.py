import numpy as np
import matplotlib.pyplot as plt
import h5py


class Feed_Forward_NN():

    def __init__(self):
        pass

    def load_dataset(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def set_hyperparameters(self, learning_rate, layer_dims, lambd = 0, dropout=1):
        self.alpha = learning_rate
        self.layers_dimension = layer_dims
        self.lambd = lambd
        self.dropout = dropout

    def initialize_parameters(self, dims, _type='he'):
        np.random.seed(3)
        parameters = {}
        L = len(dims)
        for l in range(1, L):
            if _type == 'random':
                parameters['W' + str(l)] = np.random.randn(dims[l], dims[l-1]) * 0.01
                parameters['b' + str(l)] = np.zeros([dims[l], 1])
            elif _type == 'he':
                parameters['W' + str(l)] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2/dims[l-1])
                parameters['b' + str(l)] = np.zeros([dims[l], 1])
        self.parameters = parameters
        return parameters


    def compute_cost(self, AL, Y, caches):
        m = AL.shape[1]
        
        cost = (-1.0/m) * np.sum( np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y) )
        if self.lambd != 0:
            regularization_cost = 0
            for cache in caches:
                regularization_cost += np.sum(np.sum(np.power(cache[1],2)))
            cost += (self.lambd/(2*m))*regularization_cost
             
        return np.squeeze(cost)

    def sigmoid(self, Z):
        return 1.0/(1.0 + np.exp(-Z))

    def sigmoid_gradient(self, dA, Z):
        A = self.sigmoid(Z)
        gradient = np.multiply(A, 1 - A)
        return np.multiply(dA, gradient)

    def relu(self, Z):
        return np.maximum(0, Z)


    def forward_propagation(self, X, parameters):
        
        np.random.seed(1)
        caches = []
        A = X
        L = len(parameters) // 2

        for i in range(1,L):
            A_prev = A

            # dropout
            # D = ( np.random.rand(A_prev.shape[0], A_prev.shape[1]) < self.dropout ).astype(int)
            # A_prev = np.multiply(A_prev, D)
            # A_prev = A_prev/self.dropout

            W = parameters['W' + str(i)]
            b = parameters['b' + str(i)]
            Z = np.dot(W, A_prev) + b
            A = self.relu(Z)
            # cache = [A_prev, W, b, Z, D]
            cache = [A_prev, W, b, Z]
            caches.append(cache)


        # dropout
        # D = ( np.random.rand(A.shape[0], A.shape[1]) < self.dropout ).astype(int)
        # A = np.multiply(A, D)
        # A = A/self.dropout

        WL = parameters['W' + str(L)]
        bL = parameters['b' + str(L)]
        ZL = np.dot(WL, A) + bL
        AL = self.sigmoid(ZL)
        cache = [A, WL, bL, ZL]
        caches.append(cache)

        assert(AL.shape == (1,X.shape[1]))

        return AL, caches


    def linear_backward(self, dA, cache, activation='relu'):

        if len(cache) == 4:
            A_prev, W, b, Z = cache
            D = np.ones((A_prev.shape[0], A_prev.shape[1]), dtype=float)
        else:
            A_prev, W, b, Z, D = cache

        m = A_prev.shape[1]
        dZ = Z
        if activation == 'sigmoid':
            dZ = self.sigmoid_gradient(dA, Z)
            
        if activation == 'relu':
            # A = self.relu(Z)
            # derivative = (Z >= 0).astype(int)
            # dZ = np.multiply(dA, derivative)
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0

        dW = (1/m)*np.dot(dZ, A_prev.T) + (self.lambd/m)*W
        db = (1/m)*np.sum(dZ, axis = 1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        # dA_prev = np.multiply(dA_prev, D)
        # dA_prev /= self.dropout


        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db


    def backward_propagation(self, AL, Y, caches):
        
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
        current_cache = caches[L-1]
        grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = self.linear_backward(dAL, current_cache, activation='sigmoid')

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            grads['dA' + str(l)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = self.linear_backward(grads['dA' + str(l + 1)], current_cache, activation='relu')
        return grads


    def update_parameters(self, parameters, grads, alpha):
        L = len(parameters)//2
        temp_param = parameters
        for l in range(1, L + 1):
            temp_param['W' + str(l)] = temp_param['W' + str(l)] - alpha*grads['dW' + str(l)]
            temp_param['b' + str(l)] = temp_param['b' + str(l)] - alpha*grads['db' + str(l)]
        return temp_param

    def predict(self, X_test):
        threshold = 0.5
        predictions, _ = self.forward_propagation(X_test, self.parameters)
        predictions = predictions > threshold
        return predictions.astype(int)

    def train_model(self):

        parameters = self.initialize_parameters(self.layers_dimension, _type='he')
        num_iterations = 2500
        costs = []

        for i in range(num_iterations):
            AL, caches = self.forward_propagation(self.X_train, parameters)
            cost = self.compute_cost(AL, self.Y_train, caches)
            grads = self.backward_propagation(AL, self.Y_train, caches)
            parameters = self.update_parameters(parameters, grads, self.alpha)

            if i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                costs.append(cost)
                    
            
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.alpha))
        plt.show()
        self.parameters = parameters

    def mini_batch_gradient(self):
        parameters = self.initialize_parameters(self.layers_dimension, _type='he')
        batch_size = 5000
        T = self.X_train.shape[1]//batch_size

        for t in range(T):
            X = self.X_train[:, t*batch_size:(t+1)*batch_size]
            Y = self.Y_train[0, t*batch_size:(t+1)*batch_size]

            AL, caches = self.forward_propagation(X, parameters)
            cost = self.compute_cost(AL, Y, caches)
            grads = self.backward_propagation(AL, Y, caches)
            parameters = self.update_parameters(parameters, grads, self.alpha)

    def get_parameters(self):
        return self.parameters
    
    def save_model(self):
        model = {
            'params' : self.parameters,
            'learning_rate' : self.alpha,
            'layers_dimension' : self.layers_dimension,
            'lambda' : self.lambd
        }
        with open('model.txt', 'w') as f:
            f.write(str(model))
        
    # def load_model(self,file_name):
    #     with open(file_name, 'r') as f:
    #         data = eval(str(f.read()))
    #     self.parameters = data['params']
    #     self.learning_rate = data['learning_rate']
    #     self.layers_dimension = data['layers_dimension']
    #     self.lambd = data['lambda']
    #     return data


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


NN = Feed_Forward_NN()

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

NN.set_hyperparameters(learning_rate=0.0075, layer_dims=[12288, 20, 7, 5, 1], lambd=0, dropout=0.8)
NN.load_dataset(train_x, train_y)
NN.train_model()
NN.save_model()
