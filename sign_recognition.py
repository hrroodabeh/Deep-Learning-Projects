import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import h5py
from deep_utils import generate_random_mini_batches

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x])
    Y = tf.placeholder(tf.float32, [n_y])    
    return X, Y

def initialize_parameters():
    tf.set_random_seed(1) 
        
    W1 = tf.get_variable('W1', [25, 12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable('b1', [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable('W2', [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable('b2', [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable('W3', [6, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable('b3', [6, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters








if __name__ == '__main__':
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    test_pic = X_test_orig[0]
    # X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], X_train_orig.shape[1]*
                        X_train_orig.shape[2]*X_train_orig.shape[3]).T

    # X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], X_test_orig.shape[1]*
                        X_train_orig.shape[2]*X_test_orig.shape[3]).T

    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.

    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)

    print(test_pic[0][0])