import tensorflow as tf
import numpy as np 
import math 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

#generating random house sizes
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

#generate random house price with random offset
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

#plotting data on a graph
plt.plot(house_size, house_price, "bx") #bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

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
#tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)
tf_cost = tf.reduce_mean(tf.square(tf_price_pred - tf_price))
#the size of steps down the graident
learning_rate = 0.01

#minimise loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

#initialise variables
init = tf.global_variables_initializer()

#launch the session
with tf.Session() as sess:
    sess.run(init)

    #how often to display progress and number of iterations
    display_every = 2
    num_training_iter = 200

    #keep training for number specified
    for iteration in range(num_training_iter):

        #loop over training data, zip function lines up the two arrays
        for(x,y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        #display current status
        # if(iteration + 1) % display_every == 0:
        #     c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
        #     print("iteration#:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
        #     "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))
    
    print("Training finished")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

    saver = tf.train.Saver([tf_price_offset, tf_size_factor])
    saver.save(sess, "model/TwoFeaturePrediction")

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("price")
    plt.xlabel("Size")
    plt.plot(train_house_size, train_price, 'go', label="Training data")
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
            (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
            label="Learned regression")
    #plt.plot(house_size, ((sess.run(tf_size_factor) * (house_size * 100.0)) + sess.run(tf_price_offset)), label="fitted line")
    plt.legend(loc='upper left')
    print((sess.run(tf_size_factor)* (2500 * 100)) + sess.run(tf_price_offset))

    plt.show()



