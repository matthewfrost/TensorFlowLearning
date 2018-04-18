import tensorflow as tf
import sys
import os
from tensorflow.examples.tutorials.mnist import input_data

logPath = "./tb_logs/"

def varable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#interfactive session makes it the defalt session so we dont need to pass sess
sess = tf.InteractiveSession()

with tf.name_scope("MNIST_Input"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

#adding shape to the images, each image is now a 28pxx28px x1 cube
#-1 is a flag, only one dimension can be -1 https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/reshape
with tf.name_scope("Input_Reshape"):
    x_image = tf.reshape(x, [-1,28,28,1], name="x_image")
    tf.summary.image('input_img', x_image, 5)

#functions to make sure that values are initially positive
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#functions so that settings can be changed in one place
def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#ksize is the amount of the image we process at a time
def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#1st convolution layer
#32 features for each 5x5 section of image
#5x5 pixels, 32 features, 1 input channel because greyscale (if colour, this would be 3)
with tf.name_scope('Conv1'):
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5,5,1,32], name="weight")
        varable_summaries(W_conv1)
    with tf.name_scope('bias'):
        b_conv1 = bias_variable([32], name="bias")
        varable_summaries(b_conv1)

    #do convolution on images and add bias, then pass to relu activation function
    conv1_wx_b = conv2d(x_image, W_conv1, name="conv2d") + b_conv1
    tf.summary.histogram('conv1_wx_b', conv1_wx_b)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="relu")
    tf.summary.histogram('h_conv1', h_conv1)

    #pass results to max_pool
    h_pool1 = max_pool_2x2(h_conv1, name="pool")

#2nd convolution layer
#32 input channel because we are connecting output from layer 1. adding more features
with tf.name_scope('Conv2'):
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([5,5,32,64], name="weight")
        varable_summaries(W_conv2)
    with tf.name_scope('bias'):
        b_conv2 = bias_variable([64], name="bias")
        varable_summaries(b_conv2)

    #do convolution of output of 1st layer. pool results
    conv2_wx_b = conv2d(h_pool1, W_conv2, name="conv2d") + b_conv2
    tf.summary.histogram('conv2_wx_b', conv2_wx_b)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name="relu")
    tf.summary.histogram('h_conv2', h_conv2)
    h_pool2 = max_pool_2x2(h_conv2, name="pool")

#fully connected layer
#7x7 image with 64 images to 1024 neurons
with tf.name_scope('FC'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name="weight")
    b_fc1 = bias_variable([1024], name="bias")

    #connect output of pooling layer 2 to fully connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="relu")

#drop out some neurons to prevent over training
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#readout layer - 10 neurons representing each digit 0-9
with tf.name_scope("Readout"):
    W_fc2 = weight_variable([1024,10], name="weight")
    b_fc2 = bias_variable([10], name="bias")

#define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#calculate loss
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))

#loss optimisation
#adam optimizer is variant of gradient descent that varies step size to prevent over shooting of a good solution
with tf.name_scope("loss_optimizer"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("training_accuracy", accuracy)

#merge all summaries here
summarize_all = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

#TensorBoard - write the default graph out so we can view it's structure
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

import time

num_steps = 2000
display_every = 100

start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch= mnist.train.next_batch(50)
    _, summary = sess.run([train_step, summarize_all], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    if i % display_every == 0:
        train_accuracy= accuracy.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob: 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100))
        tbWriter.add_summary(summary, i)

end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100.0))


sess.close()