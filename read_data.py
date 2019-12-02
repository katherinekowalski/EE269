import math
import os

import scipy.io
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from matplotlib import pyplot as plt

import logger


class GestureDataSequence(Sequence):
  def __init__(self, batch_size, dset="train", data_dir="data"):
    data_mat = scipy.io.loadmat(os.path.join(data_dir, "raw_data.mat"))
    if dset == "train":
      self.x, self.y = data_mat["X_train"].transpose(3,0,1,2), data_mat["Y_train"]
    elif dset == "val":
      self.x, self.y = data_mat["X_val"].transpose(3,0,1,2), data_mat["Y_val"]
    elif dset == "test":
      raise NotImplementedError
    else:
      raise ValueError

    self.x = self.x
    self.y = to_categorical(self.y - 1)
    self.batch_size = batch_size
    self.dims = self.x.shape
    self.on_epoch_end()

  def __len__(self):
    return math.ceil(self.dims[0] / self.batch_size)

  def __getitem__(self, idx):
    idxs = self.permutation[idx * self.batch_size:(idx + 1) *
    self.batch_size]
    batch_x = self.x[idxs]
    batch_y = self.y[idxs]
    return batch_x.astype(np.float32), batch_y.astype(np.float32)

  def on_epoch_end(self):
    self.permutation = np.random.permutation(range(self.dims[0]))

  def visualize(self):
    idx = self.permutation[0]
    x = self.x[idx]
    f, axes = plt.subplots(4, 4)
    for i in range(4):
      for j in range(4):
        axes[i, j].imshow(x[:, :, 4*i+j])

    plt.savefig(f"example_class{self.y[idx][0]}.png")

