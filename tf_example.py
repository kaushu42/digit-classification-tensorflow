import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load the data set from sklearn
def load_data():
    digits = load_digits()
    X = digits.data
    m = X.shape[0]
    Y = digits.target.reshape(m, 1) # Need to reshape as numpy will return a 1D array otherwise
    return X, Y

def scale(x, factor = 'auto'):
    if factor == 'auto':
        return x/x.max()
    else:
        return x/factor

def split(X, Y, ratio):
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    x, y = shuffle(X, Y)
    return train_test_split(x, y, test_size = ratio)

X, Y = load_data()
X = scale(X)
x_train, x_test, y_train, y_test = split(X, Y, 0.2)




input_size = 64
output_size = 10
hidden_size = 32





def init_params(shape):
    weights = tf.Variable(tf.random_normal(shape, stddev = 0.01))




# def forward(X, weights, biases):
#     Z1 = tf.nn.relu(tf.matmul(X, weights['W1']) + biases['b1'])
#     Y = tf.matmul(Z, weights['W2']) + biases['b2']
#     return Y

def forward(X, W1, b1, W2, b2):
    Z1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z1, W2) + b2

tfX = tf.placeholder(tf.float32, [None, input_size])
tfY = tf.placeholder(tf.float32, [None, output_size])

W1 = init_params([input_size, hidden_size])
b1 = init_params([hidden_size])
W2 = init_params([hidden_size, output_size])
b2 = init_params([output_size])

# weights = {
#             'W1': init_params([hidden_size, input_size]),
#             'W2': init_params([output_size, hidden_size])
# }
#
# biases = {
#             'b1': init_params([hidden_size, 1]),
#             'b2': init_params([output_size, 1]),
# }

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

ypred = forward(tfX, W1, b1, W2, b2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tfY, logits = ypred))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
predict_op = tf.argmax(ypred, 1)



for i in range(1000):
    feed_dict = {tfX:x_train, tfY:y_train}
    sess.run(train_op, feed_dict = feed_dict)
    pred = sess.run(predict_op, feed_dict = feed_dict)
    if i % 10 == 0:
        print(np.mean(pred == Y))
