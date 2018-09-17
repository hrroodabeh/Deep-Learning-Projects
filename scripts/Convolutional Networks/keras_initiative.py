# importing required libs and setting config variables

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import h5py
import matplotlib.pyplot as plt

import keras.backend as K
K.set_image_data_format('channels_last')
# ------------------------------------------------------------------------------------------------------ #


def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# loading datasets
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T
# ------------------------------------------------------------------------------------------------------ #


def create_model(input_shape):
    
    X_input = Input(input_shape)
      
    X = ZeroPadding2D((3, 3))(X_input)

    # Conv 32 -> Batch Normalization -> ReLU Activation
    X = Conv2D(32, (3, 3), strides = (1, 1), name='conv0')(X)
    X = BatchNormalization(axis = 3, name='bn0')(X)
    X = Activation('relu')(X)
    
    # MaxPooling with (2, 2) filter and None stride, which is equal to the pooling size (2, 2)
    X = MaxPooling2D(pool_size=(2, 2), name='mp0')(X)
    
    # flatten to use fully connected layers
    X = Flatten()(X)
    
    # a fully connected Dense layer wich is a sigmoid classifier
    X = Dense(1, activation = 'sigmoid', name='fc0')(X)
    
    model = Model(inputs= X_input, outputs=X, name='Happy_House_Model')
    
    return model


if __name__ == '__main__':

    model = create_model(X_train.shape[1:])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x = X_train, y = Y_train, epochs=7, batch_size=16)

    preds = model.evaluate(X_test, Y_test)

    predictions = model.predict(X_test)

    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    # print(model.predict(x))

