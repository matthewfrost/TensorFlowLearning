import tensorflow as tf 

price_offset = tf.Variable(-1.0, validate_shape=False, name="price_offset")
size_factor = tf.Variable(-1.0, validate_shape=False, name="size_factor")

print((0.97969216 * 2500 * 100) + 7.931055e-05)