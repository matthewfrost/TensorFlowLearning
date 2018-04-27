import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as pyplot

from matplotlib.mlab import PCA
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import StandardScaler

prices = pd.read_csv('data/stocks.csv')

prices['Date'] = pd.to_datetime(prices['Date'], infer_datetime_format=True)

prices = prices.sort_values(['Date'], ascending=[True])
prices = prices[['AAPL', 'GOOG', 'NFLX']]

returns = prices[[key for key in dict(prices.dtypes) 
    if dict(prices.dtypes)[key] in ['float64', 'int64']]].pct_change()

returns = returns[1:]

returns_arr = returns.as_matrix()[:20]

scaler = StandardScaler()
returns_arr_scaled = scaler.fit_transform(returns_arr)

results = PCA(returns_arr_scaled, standardize=False)

#begining NN

n_inputs = 3
n_hidden = 2
n_outputs = n_inputs

learning_rate = 0.01
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, n_inputs])

#activation function for linear neurons is none
hidden = tf.layers.dense(X, n_hidden) #takes data in X as input and produces 'n_hidden' outputs
outputs = tf.layers.dense(hidden, n_outputs)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

num_epoch = 10000

with tf.Session() as sess:
    init.run()

    for iteration in range(num_epoch):
        training_op.run(feed_dict={X: returns_arr_scaled})
    
    outputs_val = outputs.eval(feed_dict={X: returns_arr_scaled})
    print(outputs_val)