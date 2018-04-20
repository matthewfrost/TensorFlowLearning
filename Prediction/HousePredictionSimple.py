import tensorflow as tf
import numpy as np 
import math 

#generating random house sizes
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

#generate random house price with random offset
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)


def normalize(array):
    return (array - array.mean()) / array.std()

#70% of data used to train 
num_train_samples = math.floor(num_house * 0.7)

#training data 
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

#test data 
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
tests_price_norm = normalize(test_house_price)

#set tf placeholders
tf_house_size = tf.placeholder("float", name="house_size") #placeholders are data passed into a computational graph
tf_price = tf.placeholder("float", name="price")

#initially set weighting to a random value that will be adjusted during training
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

#loss function - mean squared error
#mean squared error - square all the errors, add it all up, find the mean
tf_cost = tf.reduce_mean(tf.square(tf_price_pred - tf_price))
#the size of steps down the graident
learning_rate = 0.0001

#minimise loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

#initialise variables
init = tf.global_variables_initializer()

#launch the session
with tf.Session() as sess:
    sess.run(init)

    #how often to display progress and number of iterations
    display_every = 2
    num_training_iter = 1000

    #keep training for number specified
    for iteration in range(num_training_iter):

        #loop over training data, zip function lines up the two arrays
        for(x,y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})
    
    print("Training finished")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

    

    saver = tf.train.Saver([tf_price_offset, tf_size_factor])
    saver.save(sess, "model/TwoFeaturePrediction")