'''
shikhar joshi, backpropogation assignment - finished 4/12/2023
the purpose of this program is to create a neural network with 1 hidden layer that recognizes 28x28 pixel handwritten digits

THE ACCURACY SHOULD BE AROUND 86-91%

SOURCES:
https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
https://victorzhou.com/blog/softmax/
https://www.tensorflow.org/datasets/catalog/mnist
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# MR ANSARI IMPORTANT PLEASE CHANGE THIS DIRECTORY TO THE PATH TO THE digit-recognizer-dataset FOLDER - it holds the dataset
data = pd.read_csv("/Users/shikharjoshi/Downloads/neuralnetwork/digit-recognizer-dataset/train.csv")

data = np.array(data)
m, n = data.shape # rows and columns
np.random.shuffle(data) # prevents overfitting, shuffle before splitting into dev and training sets

data_dev = data[0 : 1000].T # transpose the first 1000 items from data and separate it
Y_dev = data_dev[0] # first row
X_dev = data_dev[1 : n]
X_dev = X_dev / 255.

data_train = data[1000 : m].T
Y_train = data_train[0]
X_train = data_train[1 : n]
X_train = X_train / 255.

def init_params():
    W1 = np.random.rand(10, 784) - 0.5 # generate rand values between 0 and 1 for each element of array
    b1 = np.random.rand(10, 1) - 0.5 # same as above

    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    
    return W1, b1, W2, b2

# rectified linear unit activation, ReLU(x) = { (x -> x > 0), (0 -> x <= 0) }
def ReLU(Z):
    return np.maximum(Z, 0) # go through each element in Z and perform { (x -> x > 0), (0 -> x <= 0) } for Z[i]

# derivative of RelU, dRelU = { (1 -> x > 0), (0, x <= 0) }
def ReLU_deriv(Z):
    return Z > 0 # if Z > 0, it returns true which is 1. if Z isnt, returns false which is 0

# softmax activation, = e^currentnode / sum(e^allnodes)
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A # a will be between 0 and 1
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 # get the dotproduct betweem the weights[1] and x and add the bias
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2 # get dotproduct between the weights[2] and A1 and add the bias
    A2 = softmax(Z2) # perform softmax activation function on Z2 to get second activated layer

    return Z1, A1, Z2, A2

def one_hot(Y): # one hot encoding
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # creates the correctly siZed matrix filled w/ Zeros
    one_hot_Y[np.arange(Y.size), Y] = 1 # for each row, go to column (y) and set it to 1
    one_hot_Y = one_hot_Y.T # transpose it

    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y) # onehot encode Y

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) # apply weights in reverse, also multiply derivative of activation function
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1 # apply learning rate to every parameter
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0) # get the maximum value from the output layer, that's our answer

def get_accuracy(predictions, Y):
    print(predictions, Y) # print
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params() # initialize all the parameters
    for i in range(iterations): # go through all iterations
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X) # forward propogate
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) # backpropogate
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) # update all the parameters with learning rate in mind
        if i % 10 == 0: # every 10th iteration
            print("iteration: ", i) # print the iter number
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y)) # get acc
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None] # get the current data's 28x28 image
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]

    # show the prediction vs actual value
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    # matplotlib magic to show the image
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.30, 500)

# testing, make predictions by looking at individual test datas,
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

# actual testing data to prevent overfitting
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print("accuracy: ", get_accuracy(dev_predictions, Y_dev))