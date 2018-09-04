import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf

from deep_utils import generate_random_mini_batches
from sign_recognition import forward_propagation as predict_forward
from sign_recognition import load_dataset



def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = predict_forward(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction



def one_hot(Y, C):
    m = Y.shape[1]
    T = np.zeros((C, m))
    for i in range(m):
        T[Y[0, i], i] = 1
    return T



def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name= 'X')
    Y = tf.placeholder(tf.float32, [None, n_y], name= 'Y')
    return X, Y



def initialize_parameters():
    # f = 4, n_c_prev = 3, n_c = 8
    W1 = tf.get_variable('W1', [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    # f = 2, n_c_prev = 8, n_c = 16
    W2 = tf.get_variable('W1', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {
        'W1' : W1,
        'W2' : W2
    }
    return parameters


def forward_propagation(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    
    # first conv layer, with filters W1, Strides of [1 verical, 1 horizontal] and same padding
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')

    # Relu activation
    A1 = tf.nn.relu(Z1)

    # max pooling with filter size of 8 by 8, and same padding
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding= 'SAME')

    # second conv layer with strides [1, 1]
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding= 'SAME')

    A2 = tf.nn.relu(Z2)

    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    # flattening the last layer into a vector of [batch_size, k]
    P2 = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn = None)

    return Z3


if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    X_train = X_train_orig/255.
    X_test_orig = X_test_orig/255.
    Y_train = one_hot(Y_train_orig, 6).T
    Y_test = one_hot(Y_test_orig, 6).T