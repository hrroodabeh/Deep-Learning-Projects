import numpy as np
import matplotlib.pyplot as plt
import h5py


class Feed_Forward_NN():

    def __init__(self):
        pass

    def load_dataset(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def set_hyperparameters(self, learning_rate, layer_dims):
        self.alpha = learning_rate
        self.layers_dimension = layer_dims

    def initialize_parameters(self, dims):
        np.random.seed(3)
        parameters = {}
        L = len(dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(dims[l], dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros([dims[l], 1])
        self.parameters = parameters
        return parameters


    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        cost = (-1.0/m) * np.sum( np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y) )
        return np.squeeze(cost)

    def sigmoid(self, Z):
        return 1.0/(1 + np.exp(-Z))

    def sigmoid_gradient(self, dA, A):
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
            W = parameters['W' + str(i)]
            b = parameters['b' + str(i)]
            Z = np.dot(W, A) + b
            A = self.relu(Z)
            cache = [A_prev, W, b, Z]
            caches.append(cache)

        if L != 4:
            print('khaak too sater')
        WL = parameters['W' + str(L)]
        bL = parameters['b' + str(L)]
        ZL = np.dot(WL, A) + bL
        AL = self.sigmoid(ZL)
        cache = [A, WL, bL, ZL]
        caches.append(cache)

        return AL, caches


    def linear_backward(self, dA, cache, activation):
        A_prev, W, b, Z = cache
        m = A_prev.shape[1]
        dZ = Z
        if activation == 'sigmoid':
            A = self.sigmoid(Z)
            derivative = np.multiply(A, 1 - A)
            dZ = np.multiply(dA, derivative)
            
        if activation == 'relu':
            # A = self.relu(Z)
            # derivative = (Z >= 0).astype(int)
            # dZ = np.multiply(dA, derivative)
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0

        dW = (1/m)*np.dot(dZ, A_prev.T)
        db = (1/m)*np.sum(dZ, axis = 1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db


    def backward_propagation(self, AL, Y, caches):
        
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)

        dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
        current_cache = caches[L-1]
        grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = self.linear_backward(dAL, current_cache, 'sigmoid')

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            grads['dA' + str(l)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = self.linear_backward(grads['dA' + str(l + 1)], current_cache, 'relu')
        return grads


    def update_parameters(self, parameters, grads, alpha):
        L = len(parameters)//2
        temp_param = parameters
        for l in range(1, L + 1):
            temp_param['W' + str(l)] = parameters['W' + str(l)] - alpha*grads['dW' + str(l)]
            temp_param['b' + str(l)] = parameters['b' + str(l)] - alpha*grads['db' + str(l)]
        return temp_param

    def predict(self, X_test):
        threshold = 0.5
        predictions = self.forward_propagation(X_test, self.parameters)
        predictions = predictions > threshold
        return predictions.astype(int)

    def train_model(self):

        parameters = self.initialize_parameters(self.layers_dimension)
        num_iterations = 2500
        costs = []

        for i in range(num_iterations):
            AL, caches = self.forward_propagation(self.X_train, parameters)
            cost = self.compute_cost(AL, self.Y_train)
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


    def get_parameters(self):
        return self.parameters


def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
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

NN.set_hyperparameters(0.0075, [12288, 20, 7, 5, 1])
NN.load_dataset(train_x, train_y)
NN.train_model()
