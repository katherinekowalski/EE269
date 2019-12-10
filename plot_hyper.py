import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import json
import os

import logger

LOGS = "logs"
FIGURES = "figures"

# Create some mock data
def main():
  run_params = []
  results = []
  for run_name in os.listdir(LOGS):
    if run_name.split(".")[-1] != "json": continue
    run_id = run_name.split('.')[0]
    logger.info(f"plotting {run_name}")
    fn = os.path.join(LOGS, run_name)
    with open(fn, "r") as file:
      data = json.load(file)
      run_params.append(data["params"])
      results.append(max(data["val_acc"]))
      if max(data["val_acc"]) == max(results):
        best_run_id = run_id
      plot(data, run_id)

  for param in run_params[-1].keys():
    logger.info("plotting " + param)
    x = [run[param] for run in run_params if param in run]
    if type(x[0]) == list:
      x = [str(xi) for xi in x]
    plt.figure()
    if param in ["learning_rate"] or "reg" in param:
      plt.xscale('log')
    # print (x)
    # print (results)
    plt.scatter(x, results, color="blue")
    # means = []
    # for uxi in list(set(x)):
    #   mean = np.mean([r for (i, r) in enumerate(results) if run_params[i][u]])
    
    print (x)
    ux = list(set(x))
    ls = [np.mean([r for i, r in enumerate(results) if param in run_params[i] and run_params[i][param] == uxi]) for uxi in ux]
    print (ls)
    plt.scatter(ux, ls, color="green")
    plt.title(param)
    plt.xlabel(param)
    plt.ylabel("max val f2")
    plt.savefig(os.path.join(FIGURES, param+".png"))
    plt.close()

  print (f"best run: {best_run_id} had accuracy {max(results)}")
  return best_run_id



def plot(data, run_id):
  fig, ax1 = plt.subplots()
  it = list(range(len(data["val_acc"])))
  val_it = list(range(len(data["val_acc"])))

  ax2 = ax1.twinx()
  ax3 = ax2.twiny()
  ax4 = ax1.twiny()
  ax4.set_ylabel('validation iteration')

  color = 'tab:red'
  ax1.set_xlabel('train iterations')
  ax1.set_ylabel('loss', color=color)
  ltl = ax1.plot(it, data["loss"], color=color, label="train loss")
  ax1.tick_params(axis='y', labelcolor=color)

  color1 = 'tab:blue'
  color2 = 'tab:green'
  ax2.set_ylabel('pct', color=color)  # we already handled the x-label with ax1
  lta = ax2.plot(it, data["acc"], color=color1, label="train accuracy")
  # ltf = ax2.plot(it, data["train_f2"], color=color2, label="train F2")
  ax2.tick_params(axis='y', labelcolor=color)

  lva = ax3.plot(val_it, data["val_acc"], color="tab:purple", label="val accuracy")
  # lvf = ax3.plot(val_it, data["val_f2"], color="tab:orange", label="val F2")
  lvl = ax4.plot(val_it, data["val_loss"], color="black", label="val loss")

  all_lines = ltl + lta + lva + lvl
  labels = [l.get_label() for l in all_lines]
  plt.legend(all_lines, labels, loc="lower left")

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.savefig(os.path.join(FIGURES, run_id.split(".")[0] + ".png"))
  plt.close()

if __name__ == "__main__":
  main()
