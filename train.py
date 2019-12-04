import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from read_data import GestureDataSequence, get_dataset
from models import get_cnn_model_1, get_3dcnn_model_1, get_params

from collections import defaultdict
import logger
import math
import random
import json
import os
import string
import argparse

# tf.compat.v1.enable_eager_execution()

colab = False

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-dir", default="tuning_logs", action="store")
  parser.add_argument("--data-dir", default="../data", action="store")
  args_raw = parser.parse_args()
  return {
    "log_dir": args_raw.log_dir,
    "data_dir": args_raw.data_dir
  }


def train(data_dir, params, epochs=20):
  logger.info(f"run_id: {params['run_id']}")
  logger.info("getting model...")
  model = get_cnn_model_1(**params)
  logger.info("compiling model...")
  opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
  model.compile(opt, loss="categorical_crossentropy", metrics=["accuracy", "categorical_crossentropy"])
  logger.info("training...")
  # history = model.fit_generator(
  #   GestureDataSequence(params["batch_size"], dset="train", data_dir=data_dir),
  #   validation_data=GestureDataSequence(params["batch_size"], dset="val", data_dir=data_dir), 
  #   epochs=epochs)
  history = model.fit(get_dataset("train"),
                      validation_data=get_dataset("val"),
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(f"cnn1_{params['run_id']}.chkpt", save_best_only=True)],
                      validation_steps=60,
                      epochs=epochs)
  logger.info(model.summary())
  return history


def main():
  args = get_args()
  params = get_params()
  train_result = train(args["data_dir"], params)
  print (train_result.history)

def tune():
  args = get_args()
  while True:
    params = get_random_params()
    logger.info("training with parameters:")
    logger.info(params)
    train_result = train(args["data_dir"], params)
    with open(os.path.join(args["log_dir"], params["run_id"] + ".json"), "w+") as file:
      json.dump(train_result, file)

if __name__ == "__main__":
  main()
