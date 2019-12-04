import tensorflow as tf
from scipy.io import loadmat
import h5py
import numpy as np
from tqdm import tqdm 

import random
import os

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

DATA_DIR = "../data"

def write(dset="train"):
  out_filename = f"/Volumes/WINSTANLEY_SSD/{dset}____.tfrecords"
  if dset == "train":
    x_filename = 'train.mat'
    idf = "X_train_wd"
    y_fn = "ytrain.mat"
    idf_y = "Y_train_wd"
  elif dset == "val":
    x_filename = 'val.mat'
    idf = "X_val_wd"
    y_fn = "yval.mat"
    idf_y = "Y_val_wd"
  elif dset == "test":
    x_filename = 'test.mat'
    idf = "X_test_wd"
    y_fn = "ytest.mat"
    idf_y = "Y_test_wd"
  else:
    raise ValueError

  xdata = h5py.File(os.path.join(DATA_DIR, x_filename), "r")
  ydata = loadmat(os.path.join(DATA_DIR, y_fn))[idf_y][0]

  file_refs = xdata[idf]
  writer = tf.io.TFRecordWriter(out_filename)
  l = list(enumerate(file_refs))
  random.shuffle(l)
  for i, userref in tqdm(l):
    labels = ydata[i]
    examples = xdata[userref[0]]
    user_examples = examples.shape[0]
    for j in range(user_examples):
      print (labels[j], labels[j].dtype)
      print (examples[j].shape)
      feature = {'train/label': _int64_feature(labels[j]),
                 'train/image': _bytes_feature(tf.compat.as_bytes(examples[j].tostring()))}
      # Create an example protocol buffer
      example = tf.train.Example(features=tf.train.Features(feature=feature))
    
      # Serialize to string and write on the file
      writer.write(example.SerializeToString())
      
  writer.close()

def main():
  write("train")
  # write("val")
  # write("test")

if __name__ == "__main__":
  main()

