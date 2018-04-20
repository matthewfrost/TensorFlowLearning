import tensorflow as tf
import numpy as np 
import math 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from keras.models import Sequential
from keras.layers.core import Dense, Activation

#generating random house sizes
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

#generate random house price with random offset
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

#plotting data on a graph
# plt.plot(house_size, house_price, "bx") #bx = blue x
# plt.ylabel("Price")
# plt.xlabel("Size")
# plt.show()

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

# begin keras
model = Sequential()
#1 neuron, 1 value but we dont know how many will be passed, initialise weights to be random uniform distribution, linear regression
model.add(Dense( 1, input_shape=(1,), kernel_initializer='uniform', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='sgd') #sgd = stocastic gradient decsent 

model.fit(train_house_size_norm, train_price_norm, nb_epoch=300)

score = model.evaluate(test_house_size_norm, tests_price_norm)
print("\nloss on test: {0}".format(score))

