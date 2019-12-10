# figures
# filters (for wavelet and raw data)
# wavelet transformations
# Learning curves
# Confusion matrix
# Data examples

from train import *
from models import *
import plot_hyper

def parse_args():
  parser = argparse.ArgumentParser()
  # parser.add_argument("--run-id", action="store")
  parser.add_argument("--out-dir", action="store")
  return parser.parse_args()

args = parse_args()

LOGS = "logs"

def main():
  best_run_id = plot_hyper.main()
  with open(os.path.join(LOGS, f"{best_run_id}.json")) as metadatafile:
    params = json.load(metadatafile)["params"]
  model = get_cnn_model_1(**params)
  opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
  model.compile(opt, loss="categorical_crossentropy", metrics=["accuracy", "categorical_crossentropy"])
  model.build([None, 128, 128, 10])
  model.load_weights(f"cnn1_{best_run_id}.chkpt")
  filters_np = model.layers[0].weights[0].numpy()
  row_2 = int(filters_np.shape[-1]/2)
  f, axes = plt.subplots(2, row_2)
  for i in range(2):
    for j in range(row_2):
      axes[i, j].matshow(filters_np[:, :, 0, row_2*i+j])
  plt.savefig(os.path.join(args.out_dir, f"filters_{best_run_id}.png"))
  




if __name__ == '__main__':
  main()
