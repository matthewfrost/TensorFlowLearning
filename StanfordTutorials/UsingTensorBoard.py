import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #suppress warnings about CPU instructions

import tensorflow as tf 


x = tf.constant(2, name="x")
y = tf.constant(3, name="y")
z = tf.add(x, y)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    print(sess.run(z))

#run tensorboard by running the command below:
# tensorboard --logdir './graphs' --host localhost --port 6006