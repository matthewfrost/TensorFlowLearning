import tensorflow as tf
import sys
import os
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as k

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

image_rows = 28
image_cols = 28

train_images = mnist.train.images.reshape(mnist.train.images.shape[0], image_rows, image_cols, 1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], image_rows, image_cols, 1)

num_filters = 32
max_pool_size= (2,2)
conv_kernel_size=(3,3)
image_shape = (28,28,1)
num_classes = 10
drop_prob= 0.5

model = Sequential()

model.add(Convolution2D(num_filters, conv_kernel_size[0], conv_kernel_size[1], border_mode='valid',
        input_shape=image_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

model.add(Convolution2D(num_filters, conv_kernel_size[0], conv_kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(drop_prob))
model.add(Activation('softmax'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size=128
num_epoch=2

model.fit(train_images, mnist.train.labels, batch_size=batch_size, nb_epoch=num_epoch,
    verbose=1, validation_data=(test_images, mnist.test.labels))
