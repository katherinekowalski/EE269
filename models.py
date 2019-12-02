import tensorflow as tf
import numpy as np

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
