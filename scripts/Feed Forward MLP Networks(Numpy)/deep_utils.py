import numpy as np
import matplotlib.pyplot as plt
# import h5py
import math
import sklearn
import sklearn.datasets

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



def initialize_parameters(dims, _type='he'):
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
    return parameters

def compute_cost(AL, Y, caches, lambd):
    m = AL.shape[1]
    
    cost = (-1.0/m) * np.sum( np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y) )
    if lambd != 0:
        regularization_cost = 0
        for cache in caches:
            regularization_cost += np.sum(np.sum(np.power(cache[1],2)))
        cost += (lambd/(2*m))*regularization_cost
            
    return np.squeeze(cost)

def sigmoid(Z):
    return 1.0/(1.0 + np.exp(-Z))

def sigmoid_gradient(dA, Z):
    A = sigmoid(Z)
    gradient = np.multiply(A, 1 - A)
    return np.multiply(dA, gradient)

def relu(Z):
    return np.maximum(0, Z)

def relu_gradient(dA, Z):
    A = relu(Z)
    ZZ = (Z > 0).astype(int)
    return np.multiply(dA, ZZ)


def forward_propagation(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2

    for i in range(1,L):
        A_prev = A
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]
        Z = np.dot(W, A_prev) + b
        A = relu(Z)
        cache = [A_prev, W, b, Z]
        caches.append(cache)

    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    ZL = np.dot(WL, A) + bL
    AL = sigmoid(ZL)
    cache = [A, WL, bL, ZL]
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches


def linear_backward(dA, cache, activation='relu'):

    A_prev, W, b, Z = cache

    m = A_prev.shape[1]
    dZ = Z
    if activation == 'sigmoid':
        dZ = sigmoid_gradient(dA, Z)
        
    if activation == 'relu':
        # dZ = np.array(dA, copy=True)
        # dZ[Z <= 0] = 0
        dZ = relu_gradient(dA, Z)

    dW = (1/m)*np.dot(dZ, A_prev.T) #+ (lambd/m)*W
    db = (1/m)*np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)


    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
    current_cache = caches[L-1]
    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_backward(dAL, current_cache, activation='sigmoid')

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA' + str(l)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = linear_backward(grads['dA' + str(l + 1)], current_cache, activation='relu')
    return grads


def update_parameters(parameters, grads, alpha):
    L = len(parameters)//2
    temp_param = parameters
    for l in range(1, L + 1):
        temp_param['W' + str(l)] = temp_param['W' + str(l)] - alpha*grads['dW' + str(l)]
        temp_param['b' + str(l)] = temp_param['b' + str(l)] - alpha*grads['db' + str(l)]
    return temp_param

# def predict(parameters, X_test):
#     threshold = 0.5
#     predictions, _ = forward_propagation(X_test, parameters)
#     predictions = predictions > threshold
#     return predictions.astype(int)


def stochastic_gradient_descent(X, Y, alpha):
    dims = []
    parameters = initialize_parameters(dims, _type='random')
    num_iter = 5
    costs = []
    assert(X.shape[1] == Y.shape[1])

    m = X.shape[1]
    
    for _ in range(num_iter):
        for j in range(m):
            AL, caches = forward_propagation(X[:, j], parameters)
            cost = compute_cost(AL, Y[:, j], caches, 0)
            grads = backward_propagation(AL, Y[:, j], caches)
            parameters = update_parameters(parameters, grads, alpha)
            costs.append(cost)
    return parameters

def generate_random_mini_batches(X, Y, mini_batch_size=64, seed = 0):

    np.random.seed(seed)
    m = X.shape[1]

    full_mini_batch_count = math.floor(m/mini_batch_size)

    shuffle_indexing = list(np.random.permutation(m))
    shuffled_X = X[:, shuffle_indexing]
    shuffled_Y = Y[:, shuffle_indexing]

    mini_batches = []

    for i in range(full_mini_batch_count):
        mini_batch = (shuffled_X[:, i*mini_batch_size:(i+1)*mini_batch_size], shuffled_Y[:, i*mini_batch_size:(i+1)*mini_batch_size])
        mini_batches.append(mini_batch)
    if m%mini_batch_size != 0:
        mini_batch = (shuffled_X[:, full_mini_batch_count*mini_batch_size:], shuffled_Y[:, full_mini_batch_count*mini_batch_size:])
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batch_gradient_descent(X, Y, alpha, batch_size):
    dims = []
    parameters = initialize_parameters(dims, _type='random')
    num_iter = 5
    costs = []
    assert(X.shape[1] == Y.shape[1])

    mini_batches = generate_random_mini_batches(X, Y, batch_size)

    T = len(mini_batches)

    for _ in range(num_iter):
        for t in range(T):
            x = mini_batches[t][0]
            y = mini_batches[t][1]
            AL, caches = forward_propagation(x, parameters)
            cost = compute_cost(AL, y, caches, 0)
            grads = backward_propagation(AL, y, caches)
            parameters = update_parameters(parameters, grads, alpha)
            costs.append(cost)

def initialize_velocities(parameters):
    V = {}
    L = len(parameters//2)
    for l in range(1, L+1):
        V['dW' + str(l)] = np.zeros( ( parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1] ) )
        V['db' + str(l)] = np.zeros( ( parameters['W' + str(l)].shape[0], parameters['b' + str(l)].shape[1] ) )
    return V


def initilize_adam(parameters):
    V = {}
    S = {}
    L = len(parameters)//2
    for l in range(1, L+1):
        V['dW' + str(l)] = np.zeros( ( parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1] ) )
        V['db' + str(l)] = np.zeros( ( parameters['b' + str(l)].shape[0], parameters['b' + str(l)].shape[1] ) )
        assert(parameters['W' + str(l)].shape == V['dW' + str(l)].shape)
        assert(parameters['b' + str(l)].shape == V['db' + str(l)].shape)

        S['dW' + str(l)] = np.zeros( ( parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1] ) )
        S['db' + str(l)] = np.zeros( ( parameters['b' + str(l)].shape[0], parameters['b' + str(l)].shape[1] ) )
        assert(parameters['W' + str(l)].shape == S['dW' + str(l)].shape)
        assert(parameters['b' + str(l)].shape == S['db' + str(l)].shape)

    return V, S


def update_parameters_momentum(parameters, grads, learning_rate, V, momentum_beta):
    L = len(parameters)//2
    for l in range(1, L + 1):
        V['dW' + str(l)] = momentum_beta*V['dW' + str(l)] + (1 - momentum_beta)*grads['dW' + str(l)]
        V['db' + str(l)] = momentum_beta*V['db' + str(l)] + (1 - momentum_beta)*grads['db' + str(l)]

        parameters['dW' + str(l)] -= learning_rate*V['dW' + str(l)]
        parameters['db' + str(l)] -= learning_rate*V['db' + str(l)]
    return parameters, V

def update_parameters_adam(parameters, grads,  V, S, t, learning_rate=0.01, momentum_beta=0.9, RMSprop_beta=0.999, epsilon=1e-8):
    L = len(parameters)//2
    V_corrected = {}
    S_corrected = {}
    for l in range(1, L + 1):
        V['dW' + str(l)] = momentum_beta*V['dW' + str(l)] + (1 - momentum_beta) * grads['dW' + str(l)]
        V['db' + str(l)] = momentum_beta*V['db' + str(l)] + (1 - momentum_beta) * grads['db' + str(l)]

        V_corrected['dW' + str(l)] = V['dW' + str(l)]/(1 - np.power(momentum_beta, t))
        V_corrected['db' + str(l)] = V['db' + str(l)]/(1 - np.power(momentum_beta, t))

        S['dW' + str(l)] = RMSprop_beta*S['dW' + str(l)] + (1 - RMSprop_beta) * np.power(grads['dW' + str(l)], 2)
        S['db' + str(l)] = RMSprop_beta*S['db' + str(l)] + (1 - RMSprop_beta) * np.power(grads['db' + str(l)], 2)

        S_corrected['dW' + str(l)] = np.divide(S['dW' + str(l)], (1 - np.power(RMSprop_beta, t)))
        S_corrected['db' + str(l)] = np.divide(S['db' + str(l)], (1 - np.power(RMSprop_beta, t)))

        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * V_corrected['dW' + str(l)]/(np.sqrt(S_corrected['dW' + str(l)] + epsilon))
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * V_corrected['db' + str(l)]/(np.sqrt(S_corrected['db' + str(l)] + epsilon))

    return parameters, V, S


        
    

def neural_net_model(X, Y, layer_dimensions, optimizer, learning_rate=0.0007, epochs_nums=10000,
                     momentum_beta=0.9, RMSprop_beta=0.999, mini_batch_size=64, epsilon=1e-8, lambd=0):

    L = len(layer_dimensions) - 1
    parameters = initialize_parameters(layer_dimensions, _type='he')
    costs = []
    seed = 10
    t = 1

    if optimizer == 'gradient_descent':
        pass
    elif optimizer == 'momentum':
        V = initialize_velocities(parameters)
    elif optimizer == 'adam':
        V, S = initilize_adam(parameters) 
    else:
        pass

    for i in range(epochs_nums):
        random_seed = np.random.randint(20)
        np.random.seed(random_seed)
        # seed = seed + 1
        mini_batches = generate_random_mini_batches(X, Y, mini_batch_size, seed)

        for mini_batch in mini_batches:
            x, y = mini_batch
            AL, caches = forward_propagation(x, parameters)
            cost = compute_cost(AL, y, caches, lambd)
            grads = backward_propagation(AL, y, caches)

            if optimizer == 'gradient_descent':
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == 'momentum':
                parameters, V = update_parameters_momentum(parameters, grads, learning_rate, V, momentum_beta)
            elif optimizer == 'adam':
                parameters, V, S = update_parameters_adam(parameters, grads, V, S, t, learning_rate, momentum_beta, RMSprop_beta, epsilon)
                t = t + 1

        if i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if i % 100 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()


    return parameters







if __name__ == '__main__':

    def plot_decision_boundary(model, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
        plt.show()

    def predict(X, y, parameters):
        m = X.shape[1]
        p = np.zeros((1,m), dtype = np.int)
        a3, caches = forward_propagation(X, parameters)
        for i in range(0, a3.shape[1]):
            if a3[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
        
        return p

    def predict_dec(parameters, X):
        a3, _ = forward_propagation(X, parameters)
        predictions = (a3 > 0.5)
        return predictions


    def load_dataset():
        np.random.seed(3)
        train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
        # plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
        train_X = train_X.T
        train_Y = train_Y.reshape((1, train_Y.shape[0]))
        
        return train_X, train_Y

    train_X, train_Y = load_dataset()

    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = neural_net_model(train_X, train_Y, layers_dims, optimizer = "adam")

    # Predict
    predictions = predict(train_X, train_Y, parameters)

    # Plot decision boundary
    plt.title("Model with Gradient Descent optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)