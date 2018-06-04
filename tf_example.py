import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the data set from sklearn
def load_data():
    digits = load_digits()
    X = digits.data
    m = X.shape[0]
    Y = digits.target.reshape(m, 1) # Need to reshape as numpy will return a 1D array otherwise
    return X, Y

# Use this if you want to scale the image
def scale(x, factor = 'auto'):
    if factor == 'auto':
        return x/x.max()
    else:
        return x/factor

# Randomly shuffle the set and split it into test and train set
def split(X, Y, ratio):
    x, y = shuffle(X, Y)
    return train_test_split(x, y, test_size = ratio)

# Perform one hot encoding on the labels
def encode(z):
    onehot = OneHotEncoder()
    return onehot.fit_transform(z).toarray()


D = 64 # dimensionality of input
M = 32 # hidden layer size
K = 10 # number of classes


# tensor flow variables are not the same as regular Python variables
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def forward(X, W1, b1, W2, b2):
    Z = tf.nn.relu(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2

def main():
    X, Y = load_data()
    x_train, x_test, y_train, y_test = split(X, Y, 0.3)

    y_train_org = y_train.copy().T[0] # Creating a copy of train data without one hot encoding. It will be used in checking the predictions
    y_test_org = y_test.copy().T[0] # Creating a copy of test data without one hot encoding. It will be used in checking the predictions

    y_train = encode(y_train)
    y_test = encode(y_test)


    tfX = tf.placeholder(tf.float32, [None, D])
    tfY = tf.placeholder(tf.float32, [None, K])

    W1 = init_weights([D, M]) # create symbolic variables
    b1 = init_weights([M])
    W2 = init_weights([M, K])
    b2 = init_weights([K])

    ypred = forward(tfX, W1, b1, W2, b2)

    # It performs unscaled predictions. It internally performs softmax on the predictions. If used with softmax externally too, it will perform incorrectly.
    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=ypred)
            )

    # The training operation is gradient descent with a learning rate of 0.05 to minimize the cost
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an

    # Converts the predictions(probabilities) to class values(0, 1, ..., 9). The other input is the axis on which to perform max operation
    predict_op = tf.argmax(ypred, 1)

    # Create a session and initialize the tensorflow variables
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Start the training
    for i in range(500):
        sess.run(train_op, feed_dict={tfX: x_train, tfY: y_train})
        pred_train = sess.run(predict_op, feed_dict={tfX: x_train, tfY: y_train})
        pred_test = sess.run(predict_op, feed_dict={tfX: x_test, tfY: y_test})
        if i % 100 == 0:
            print("Train Accuracy:", np.mean(y_train_org == pred_train))
            print("Test Accuracy:", np.mean(y_test_org == pred_test))
            print()



if __name__ == '__main__':
    main()
