import tensorflow as tf
import numpy as np

import string, random

def get_params():
  return {
    "dense_reg": 1e-3,
    "kernel_reg": 1e-3,
    "learning_rate": 7.5e-4,
    "kernel_sizes": [3, 5, 3, 3, 3],
    "pool_sizes": [2,2,4,2,1],
    "layer_channels": [8, 12, 12, 12, 14], 
    # "layer_channels": [3,4,5,6],
    "batch_size": 32,
    "run_id": ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)]),
  }

def get_cnn_model_1(**params):
  layer_list = []
  for i in range(len(params["kernel_sizes"])):
    ks = params["kernel_sizes"][i]
    ps = params["pool_sizes"][i]
    next_conv = tf.keras.layers.Conv2D(params["layer_channels"][i], (ks, ks), padding="SAME", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(params["kernel_reg"]))
    next_pool = tf.keras.layers.MaxPool2D((ps, ps))
    layer_list.append(next_conv)
    if ps > 1:
      layer_list.append(next_pool)
  layer_list.append(tf.keras.layers.Flatten())
  layer_list.append(tf.keras.layers.Dense(11, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(params["dense_reg"])))
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
  layer_list.append(tf.keras.layers.Dense(11, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(params["dense_reg"])))
  return tf.keras.Sequential(layer_list)
