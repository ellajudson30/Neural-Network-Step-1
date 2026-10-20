import os
import subprocess
import sys
from nn_model import ENSEMBLE_SEEDS

# Train one network per seed in nn_model.ENSEMBLE_SEEDS (in parallel) with cross_valid.py.
# Each run writes model_weights_s1_seed<seed>.pth; all runs use the same train/test split,
# so they write the same standardization_NN3.npz. Plots are switched off.

def train_ensemble(seeds=ENSEMBLE_SEEDS):
    env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="2")
    runs = []
    for seed in seeds:
        log = open(f"train_seed{seed}.log", "w")
        runs.append((seed, log, subprocess.Popen([sys.executable, "cross_valid.py", str(seed)],
                                                 stdout=log, stderr=subprocess.STDOUT, env=env)))
    for seed, log, run in runs:
        run.wait()
        log.close()
        if run.returncode != 0:
            raise RuntimeError(f"training with seed {seed} failed, see train_seed{seed}.log")
        print(f"seed {seed} trained")

if __name__ == "__main__":
    train_ensemble()
