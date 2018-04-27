import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt 


num_points = 100
dimensions = 2
points = np.random.uniform(0, 1000, [num_points, dimensions])

def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

plt.scatter(points, np.zeros_like(points), s=500)
plt.show()
num_clusters=2
kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_clusters, use_mini_batch=False)

num_iterations=10
previous_centers=None

for _ in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  previous_centers = cluster_centers
  #print('score:', kmeans.score(input_fn_1d))
print('cluster centers:', cluster_centers)
