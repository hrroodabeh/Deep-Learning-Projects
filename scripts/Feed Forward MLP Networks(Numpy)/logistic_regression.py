
import numpy as np
import matplotlib.pyplot as plt
import h5py
    
    
def load_dataset():
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

class Logistic_Regression():

    def __init__(self):
        pass

    def load_dataset(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
        self.num_features = x_train.shape[0]
        self.training_size = x_train.shape[1]
        self.params = {}

    def sigmoid(self, Z):
        return 1.0/(1 + np.exp(-Z))

    def init_weigths(self, n):
        params = {}
        params['w'] = np.zeros([n, 1])
        params['b'] = 0
        return params

    def compute_cost(self, X, Y, params):
        w = params['w']
        b = params['b']
        Z = np.dot(w.T, X) + b
        A = self.sigmoid(Z)
        m = X.shape[1]
        # cost = (-1.0/m) * np.sum( np.multiply( Y, np.log(A)) + np.multiply( 1 - Y, np.log(1 - A) ) )
        cost = (-1.0/m) * np.sum( np.multiply( np.log( A ), Y ) + np.multiply( 1 - Y, np.log( 1 - A ) ) )                            

        dw = (1.0/m)*np.dot(X, (A - Y).T)
        db = (1.0/m)*np.sum(A - Y)

        grads = {}
        grads['dw'] = dw
        grads['db'] = db
        return grads, cost

    def update_params(self, params, grads, alpha):
        params['w'] -= alpha*grads['dw']
        params['b'] -= alpha*grads['db']
        return params

    def predict(self, x_test):
        w = self.params['w']
        b = self.params['b']
        p = self.sigmoid(np.dot(w.T, x_test ) + b)
        p = p > 0.5
        return p.astype(int)

    def train_model(self):
        params = self.init_weigths(self.num_features)
        num_iters = 2000
        alpha = 0.5
        costs = []
        for _ in range(num_iters):
            grads, cost = self.compute_cost(self.x_train, self.y_train, params)
            costs.append(cost)
            params = self.update_params(params, grads, alpha)
        self.params = params


        Y_prediction_test = self.predict(self.x_test)
        Y_prediction_train = self.predict(self.x_train)
        
        d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : params['w'], 
         "b" : params['b'],
         "learning_rate" : alpha,
         "num_iterations": num_iters}

        return d






train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],
                                            train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*
                                            train_set_x_orig.shape[3]).T

test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],
                                            test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*
                                            test_set_x_orig.shape[3]).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


lr = Logistic_Regression()
lr.load_dataset(train_set_x, train_set_y, test_set_x, test_set_y)
d = lr.train_model()

index = 19
num_px = np.shape(train_set_x_orig)[1]
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
plt.show()
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")