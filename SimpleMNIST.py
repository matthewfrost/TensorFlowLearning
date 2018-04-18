import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#using tensor flow helpers to get data from MNIST site
#first param is the folder where the data will be stored
#gets the digit in 'one-hot' format, kind of like binary but only one digit is 1. e.g 0 = 10000000000, 2 = 00100000000, 5 = 00000100000
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#tensor of type float32, shape none means we dimension exists but we don't know how big
#784 because each image is 28x28px
x = tf.placeholder(tf.float32, shape=[None, 784])

#this contains the probabilty of an image being one of each digit e.g
#[0.14, 0.8, 0,0,0,0,0,0,0,0,0.06] so the image would most likely be 1
y_ = tf.placeholder(tf.float32, [None, 10])

#defined as variables because they change as the model learns
#10 neurons because we are looking at digits 0-9
W = tf.Variable(tf.zeros([784, 10])) #weight
b = tf.Variable(tf.zeros([10])) #bias

#define model, digit prediction
#value multipled by each of the weights + bias
y = tf.nn.softmax(tf.matmul(x, W) + b)

#calculate loss/cross entropy
#softmax_cross_entropy_with_logits is difference between estimated values and the actual data
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

#gradient decent to minimise loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#initialise variables
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(50000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  #gets 100 random data points from the data. batch_xs = image
                                                      #batch_ys = actual digit  
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys}) #train with this data

#comparing our prediction with the actual value
#argmax 1, gets the highest probability value
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

print("Test Accuract: {0}%".format(test_accuracy * 100.0))

sess.close()