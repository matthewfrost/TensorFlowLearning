import tensorflow as tf

x = tf.constant([2,2], name="x")
y = tf.constant([[0,1],[2,3]], name="y")
z = tf.multiply(x,y, name="z")


zeros = tf.zeros([2,3], tf.int32) #creates a tensor of the given shape filled with 0s

#creates a tensor of the same shape and type but filled with 0s
input = tf.constant([[0, 1], [2, 3], [4, 5]], name ="input")
zero_like = tf.zeros_like(input)

#same as zeros_like but 1 instead of 0
ones = tf.ones([2,2], dtype=tf.int32)
ones_like = tf.ones_like(input)

#creates a tensor of the given shape, filled with the given value
value_fill = tf.fill([2,2], 7)

#creates a 1d tensor filled with values from start to end, of the size given, can only be floats
seq = tf.lin_space(1.0,10.0, 5)

with tf.Session() as sess:
    print(sess.run(z))
    print(sess.run(seq))