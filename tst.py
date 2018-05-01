from tensorflow.python.ops.gen_nn_ops import relu

from plant_rec.dataPrep import *
import tensorflow as tf
import math
import numpy as np
import h5py
from tensorflow.python.framework import ops

# create dataset for first time
# dot_h5_make('/home/erandaka/My Studies/Sem8/Prj/Data_Set/train_images/train_dataset.h5','/home/erandaka/My Studies/Sem8/Prj/Data_Set/train_images/*.jpg')
# dot_h5_make('/home/erandaka/My Studies/Sem8/Prj/Data_Set/test_images/test_dataset.h5','/home/erandaka/My Studies/Sem8/Prj/Data_Set/test_images/*.jpg')

# load the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()
# train_dataset = load_dataset('/home/erandaka/My Studies/Sem8/Prj/Data_Set/train_images/train_dataset.h5')
# X_train_orig = train_dataset["train_x"]
# Y_train_orig = train_dataset["train_y"]
#
# test_dataset = load_dataset('/home/erandaka/My Studies/Sem8/Prj/Data_Set/test_images/test_dataset.h5')
# X_test_orig = test_dataset["train_x"]
# Y_test_orig = test_dataset["train_y"]


X_train = X_train_orig / 255
X_test = X_test_orig / 255
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T


# create placeholders
def crate_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, n_y))

    return X, Y


# initialize parameters
def initialize_parameters():
    W1 = tf.get_variable("W1", [11, 11, 3, 96], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [5, 5, 96, 256], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [3, 3, 256, 384], initializer=tf.contrib.layers.xavier_initializer())
    W4 = tf.get_variable("W4", [3, 3, 384, 384], initializer=tf.contrib.layers.xavier_initializer())
    W5 = tf.get_variable("W5", [3, 3, 384, 256], initializer=tf.contrib.layers.xavier_initializer())
    W8 = tf.get_variable("W8", [3, 3, 384, 256], initializer=tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "W5": W5}

    return parameters


# forward propagation
def forward_propagation(X, parameters):
    # alexnetarchitec:
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    W4 = parameters["W4"]
    W5 = parameters["W5"]

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 4, 4, 1], padding='VALID')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding="SAME")
    A3 = tf.nn.relu(Z3)
    Z4 = tf.nn.conv2d(A3, W4, strides=[1, 1, 1, 1], padding="SAME")
    A4 = tf.nn.relu(Z4)
    Z5 = tf.nn.conv2d(A4, W5, strides=[1, 1, 1, 1], padding="SAME")
    A5 = tf.nn.relu(Z5)
    P5 = tf.nn.max_pool(A5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    P5 = tf.contrib.layers.flatten(P5)
    Z6 = tf.contrib.layers.fully_connected(P5, 4096, activation_fn=tf.nn.relu)
    Z7 = tf.contrib.layers.fully_connected(Z6, 4096, activation_fn=tf.nn.relu)
    Z8 = tf.contrib.layers.fully_connected(Z7, 6, activation_fn=None)

    return Z8


# compute the cost
def cost_compute(Z8, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z8, labels=Y))
    return cost


# model
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.001,
          num_epochs=2, minibatch_size=150, print_cost=True):
    ops.reset_default_graph()
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = crate_placeholders(X_train.shape[1], X_train.shape[2], X_train.shape[3], Y_train.shape[1])
    parameters = initialize_parameters()
    Z8 = forward_propagation(X, parameters)
    cost = cost_compute(Z8, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    predict_op = tf.argmax(Z8, 1)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        tf.add_to_collection('predict_op', predict_op)
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions

        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        #saving parameters and accuracy values
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        saver.save(sess,'/home/erandaka/PycharmProjects/disease_yieldrec/plant_rec/modelparameters/hydroponic_garden/')

        return train_accuracy, test_accuracy, parameters


_, _, parameters = model(X_train, Y_train, X_test, Y_test)
