import tensorflow as tf
from scipy.io import loadmat
import h5py
import numpy as np
from tqdm import tqdm

import random
import threading
import os

import logger

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

DATA_DIR = "/home/colewinstanley/EE269/in_data"
OUT_DIR = "/home/colewinstanley/EE269/tfrecords_data"

def write(dset="train", half_precision=True):
  out_filename = os.path.join(OUT_DIR, f"{dset}{'_half_prec' if half_precision else ''}_no_other_raw.tfrecords")
  if os.path.exists(out_filename):
    logger.warn(f"skipping: {dset} because the .tfrecords file already exists.")
    return
  if dset == "train":
    idf = "X_train_wd"
    idf_y = "Y_train_wd"
    # filenames = ['train_0.mat', 'train_1.mat', 'train_2.mat']
    filenames = ['train_raw.mat']
  elif dset == "val":
    idf = "X_val_wd"
    idf_y = "Y_val_wd"
    filenames = ['val_raw.mat']
  elif dset == "test":
    idf = "X_test_wd"
    idf_y = "Y_test_wd"
    filenames = ['test_raw.mat']
  else:
    raise ValueError

  try:
    writer = tf.io.TFRecordWriter(out_filename)
    writer_lock = threading.Lock()

    def write_(fn):
      data = h5py.File(os.path.join(DATA_DIR, fn), "r")

      file_refs = data[idf]
      l = list(enumerate(file_refs))
      random.shuffle(l)
      for i, userref in tqdm(l):
        labels = data[data[idf_y][i][0]]
        examples = data[userref[0]]
        user_examples = examples.shape[0]
        for j in range(user_examples):
          if int(labels[0][j]) == 11:
            continue
          if half_precision:
            ex = examples[j].astype(np.float32)
          else:
            ex = examples[j]
          feature = {'train/label': _int64_feature(int(labels[0][j])),
                     'train/image': _bytes_feature(tf.compat.as_bytes(ex.tostring()))}
          # Create an example protocol buffer
          example = tf.train.Example(features=tf.train.Features(feature=feature))
        
          # Serialize to string and write on the file
          serialized = example.SerializeToString()
          writer_lock.acquire()
          writer.write(serialized)
          writer_lock.release()

    threads = []
    for i, filename in enumerate(filenames):
      logger.info(f"{dset}: reading .mat file {i+1} of {len(filenames)}.")
      threads.append(threading.Thread(target=write_, args=(filename,)))

    for thr in threads:
      thr.start()

    for thr in threads:
      thr.join()

  except:
    os.remove(out_filename)
    raise
  finally:
    writer.close()

def main():
  write("train")
  write("val")
  write("test")

if __name__ == "__main__":
  main()

