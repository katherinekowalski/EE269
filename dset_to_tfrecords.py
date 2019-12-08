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

DATA_DIR = "/home/colewinstanley/269/in_data"
OUT_DIR = "/home/colewinstanley/269/tfrecords_data"

def write(dset="train"):
  out_filename = os.path.join(OUT_DIR, f"{dset}.tfrecords")
  if dset == "train":
    filenames = ['train_0.mat', 'train_1.mat', 'train_2.mat']
  elif dset == "val":
    filenames = ['val.mat']
  elif dset == "test":
    filenames = ['test.mat']
  else:
    raise ValueError

  idf = "X_train_wd"
  idf_y = "Y_train_wd"

  writer = tf.io.TFRecordWriter(out_filename)

  for filename in filenames:
    data = h5py.File(os.path.join(DATA_DIR, filename), "r")
    # ydata = h5py.File(os.path.join(DATA_DIR, y_fn))

    file_refs = data[idf]
    l = list(enumerate(file_refs))
    random.shuffle(l)
    for i, userref in tqdm(l):
      labels = data[data[idf_y][i][0]]
      examples = data[userref[0]]
      user_examples = examples.shape[0]
      for j in range(user_examples):
        feature = {'train/label': _int64_feature(int(labels[0][j])),
                   'train/image': _bytes_feature(tf.compat.as_bytes(examples[j].tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
      
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
      
  writer.close()

def main():
  # write("train")
  write("val")
  write("test")

if __name__ == "__main__":
  main()

