import math
import os

import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from matplotlib import pyplot as plt
from tqdm import tqdm
import h5py

import logger


class GestureDataSequence(Sequence):
  def __init__(self, batch_size, dset="train", data_dir="../data"):
    # data_mat = scipy.io.loadmat(os.path.join(data_dir, "raw_data_new.mat"))
    if dset == "train":
      filename = 'train.mat'
      idf = "X_train_wd"
      y_fn = "ytrain.mat"
      idf_y = "Y_train_wd"
    elif dset == "val":
      filename = 'val.mat'
      idf = "X_val_wd"
      y_fn = "yval.mat"
      idf_y = "Y_val_wd"
    elif dset == "test":
      raise NotImplementedError
      filename = 'test.mat'
      idf = "X_test_wd"
      y_fn = "ytest.mat"
      idf_y = "Y_test_wd"
    else:
      raise ValueError

    self.f = h5py.File(os.path.join(data_dir, filename) ,'r')

    file_refs = self.f[idf]
    y = scipy.io.loadmat(os.path.join(data_dir, y_fn))[idf_y][0]
    self.example_refs = []
    for i, userref in enumerate(tqdm(file_refs)):
      labels = y[i]
      examples = self.f[userref[0]]
      user_examples = examples.shape[0]
      # TODO shouldn't need to index into y labels
      self.example_refs += [(examples[j], j, labels[j]-1) for j in range(user_examples)]

    # # self.x = self.x
    # self.y_ds_format = self.y
    # self.y = to_categorical(self.y - 1)
    self.batch_size = batch_size
    self.dim = len(self.example_refs)
    self.on_epoch_end()

  def __len__(self):
    return math.ceil(self.dim / self.batch_size)

  def __getitem__(self, idx):
    idxs = self.permutation[idx * self.batch_size:(idx + 1) *
    self.batch_size]
    batch_examples = np.stack([self.example_refs[i][0][0][self.example_refs[i][1]] for i in idxs])
    batch_y = np.stack([self.example_refs[i][2] for i in idxs])
    batch_y = to_categorical(batch_y, num_classes=11)
    return batch_examples.astype(np.float32), np.array(batch_y, dtype=np.float32)

  def on_epoch_end(self):
    self.permutation = np.random.permutation(range(self.dim))

  def visualize(self):
    x, y = self[0]
    logger.info(x.shape)
    f, axes = plt.subplots(2, 5)
    for i in range(2):
      for j in range(5):
        axes[i, j].imshow(x[0][4*i+j])

    plt.savefig(f"example_class{np.where(y[0])[0][0]}.png")

def _parse_function(proto):
    keys_to_features = {'train/image': tf.io.FixedLenFeature([], tf.string),
                        "train/label": tf.io.FixedLenFeature([], tf.int64)}
    
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
    parsed_features['train/image'] = tf.decode_raw(parsed_features['train/image'], tf.float64)
    parsed_features['train/image'] = tf.cast(parsed_features['train/image'], tf.float32)
    parsed_features['train/image'] = tf.reshape(parsed_features['train/image'],[10, 128, 128])
    parsed_features['train/image'] = tf.transpose(parsed_features['train/image'], [1,2,0])
    parsed_features['train/label'] = tf.one_hot(parsed_features['train/label']-1, depth=11)


    return parsed_features['train/image'], parsed_features["train/label"]

def get_dataset(dset="train", batch_size=32):
  filepath = "E:\\val.tfrecords"
  dataset = tf.data.TFRecordDataset(filepath)
    
  # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
  dataset = dataset.map(_parse_function, num_parallel_calls=8)
  dataset = dataset.shuffle(4000)
  dataset = dataset.batch(batch_size)
  return dataset



