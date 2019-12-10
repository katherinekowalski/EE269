import tensorflow as tf
import numpy as np

import string, random

N_CLASSES = 10

def get_params():
  return {
    "dense_reg": 5e-2,
    "kernel_reg": 5e-2,
    "learning_rate": 7.5e-4,
    "initial_kernels_sizes": [5, 3],
    "pool_sizes": [2,2,4,2,1],
    "lcc": 1, 
    # "layer_channels": [3,4,5,6],
    "batch_size": 64,
    "run_id": ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)]),
  }

def get_random_params():
  return {
    "dense_reg": 10**-(1 + random.random()*4),
    "kernel_reg": 10**-(1 + random.random()*4),
    "learning_rate": 10**-(2.25 + random.random()*2.25),
    "initial_kernels_size": random.choice([[3,3], [3,5], [5,3], [5,5]]),
    "pool_sizes": random.choice([[2,4,2,2,1], [2,2,4,2,1], [2,2,2,4,1]]),
    "lcc": 0.75 + (random.random() * 4.3), 
    "include_last_layer": random.choice([True,False]),
    "batch_size": 64,
    "run_id": ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)]),
  }

def get_cnn_model_1(**params):
  layer_list = []
  params["layer_channels"] = (np.array([8, 12, 12, 12, 14])*params["lcc"]).astype(np.int8)
  params["kernel_sizes"] = params["initial_kernels_size"] + [3,3,3]
  for i in range(len(params["kernel_sizes"])):
    if i == len(params["kernel_sizes"]) - 1 and not params["include_last_layer"]:
      continue
    ks = params["kernel_sizes"][i]
    ps = params["pool_sizes"][i]
    next_conv = tf.keras.layers.Conv2D(params["layer_channels"][i], (ks, ks), padding="SAME", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(params["kernel_reg"]))
    next_pool = tf.keras.layers.MaxPool2D((ps, ps))
    layer_list.append(next_conv)
    if ps > 1:
      layer_list.append(next_pool)
  layer_list.append(tf.keras.layers.Flatten())
  layer_list.append(tf.keras.layers.Dense(N_CLASSES, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(params["dense_reg"])))
  return tf.keras.Sequential(layer_list)

def get_3dcnn_model_1(**params):
  layer_list = []
  for i in range(len(params["kernel_sizes"])):
    ks = params["kernel_sizes"][i]
    ps = params["pool_sizes"][i]
    next_conv = tf.keras.layers.Conv3D(params["layer_channels"][i], (ks, ks, ks), padding="SAME", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(params["kernel_reg"]))
    next_pool = tf.keras.layers.MaxPool3D((ps, ps, 1))
    layer_list.append(next_conv)
    if ps > 1:
      layer_list.append(next_pool)
  layer_list.append(tf.keras.layers.Flatten())
  layer_list.append(tf.keras.layers.Dense(N_CLASSES, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(params["dense_reg"])))
  return tf.keras.Sequential(layer_list)
