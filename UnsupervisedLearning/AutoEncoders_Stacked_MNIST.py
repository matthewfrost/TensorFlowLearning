import numpy as np 
import tensorflow as tf 
import sys 
import matplotlib   
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial

mnist = input_data.read_data_sets("mnist_data/")

def display_digit(digit):
    plt.imshow(digit.reshape(28,28), cmap="Greys", interpolation="nearest")

def show_reconstructed(X, outputs, model_path= None, num_digits=2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[5 : 5 + num_digits]

        outputs_val = outputs.eval(feed_dict={X: X_test})
    fig = plt.figure(figsize=(8,3 * num_digits))

    for i in range(num_digits):
        plt.subplot(num_digits, 2,i * 2 + 1)
        display_digit(X_test[i])
        plt.subplot(num_digits, 2, i * 2 + 2)
        display_digit(outputs_val[i])
    plt.show()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

he_init = tf.contrib.layers.variance_scaling_initializer()

l2_regularizer = tf.contrib.layers.l2_regularizer(0.0001)

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

dense_layer = partial(tf.layers.dense,
                        activation=tf.nn.elu,
                        kernel_initializer=he_init,
                        kernel_regularizer=l2_regularizer)

hidden1 = dense_layer(X, n_hidden1)
hidden2 = dense_layer(hidden1, n_hidden2)
hidden3 = dense_layer(hidden2, n_hidden3)

outputs = dense_layer(hidden3, n_outputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(0.01)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 6
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        print("Running epoch: ", epoch)
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, _ = mnist.train.next_batch(batch_size) 

            sess.run(training_op, feed_dict={X: X_batch})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})

        print("\r{}".format(epoch), "Train MSE:", loss_train)

        saver.save(sess, "./stacked_autoencoder.ckpt")
show_reconstructed(X, outputs, "./stacked_autoencoder.ckpt")

