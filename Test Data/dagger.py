import csv
import numpy as np
import torch
import generate_data as g
from rk45utils import *
from nn_model import load_ensemble
from train_ensemble import train_ensemble

# DAgger for stage 1
#------------------------------------------------------------------------
# The network is trained on states the PI controller visits, so when it drives the
# solver it reaches states it has never seen (and its mistakes build up). Each
# iteration: let the trained network drive the solver on the training ODEs, label
# every state it reaches with what the PI controller would do there (the step PI would
# accept, including any retries after rejected trial steps), add those examples to the
# dataset and retrain.
# The network driving the rollouts is the seed ensemble (nn_model.py).
# The error estimate is only used for labels, never as a network input.
#
# Run after generate_data.py and train_ensemble.py. Appends to the dataset csv files,
# so rerun generate_data.py to start again from the PI-only dataset.

N_ITERATIONS = 2           # a third round did not help in testing
TOLS = [1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
MAX_STEPS = 3000           # per rollout
MAX_PER_TRAJECTORY = 30    # states kept per rollout (random sample)
rng = np.random.default_rng(1)

# Training ODEs: (name, rhs, params, initial conditions, time span) - names and params
# match generate_data.py's metadata, so new rows share their trajectory's group
JOBS = []
JOBS += [("exp decay", f, {"lambda": l}, g.ed_y0, g.ed_t_span) for l, f in zip(g.lmda, g.expdec_fcns)]
JOBS += [("logistic", f, p, y0, span) for f, p, y0, span in g.log_fcns]
JOBS += [("multi-scales", f, p, g.ms_y0, g.ms_t_span) for f, p in g.ms_fcns]
JOBS += [("oscillating", f, p, g.o_y0, g.o_t_span) for f, p in g.osc_fcns]
JOBS += [("shm", f, p, g.shm_y0, g.shm_t_span) for f, p in g.shm_fcns]
JOBS += [("vanderpol", f, p, g.vdp_y0, g.vdp_t_span) for f, p in g.VDP_fcns]
JOBS += [("damped hm", f, p, g.d_y0, g.d_t_span) for f, p in g.d_fcns]
JOBS += [("1d blows up", g.blows_up, {}, g.bu_y0, g.bu_t_span)]
JOBS += [("lorenz", f, p, g.L_y0, g.L_t_span) for f, p in g.L_fcns]
JOBS += [("chirp", f, p, g.chirp_y0, g.chirp_t_span) for f, p in g.chirp_fcns]
JOBS += [("gaussian pulse", f, p, g.pulse_y0, g.pulse_t_span) for f, p in g.pulse_fcns]
JOBS += [("duffing", f, p, g.duff_y0, g.duff_t_span) for f, p in g.duff_fcns]

def load_model():
    model = load_ensemble()
    stats = np.load('standardization_NN3.npz')
    return model, stats['mean'], stats['std']

def rollout(fcn, t0, y0, tf, tol, model, train_mean, train_std):
    """Run RK45 with the network choosing every step after a short PI start (no
    rejections, as in implement_nn.run_solverNN). Returns, for each state where the
    network chose a step: the feature vector, PI's response there, the step index,
    time and solution value."""
    c = most_varying_component(fcn, t0, y0, tf, tol)
    solver = RK45_counting(fcn, t0, y0, tf, rtol=tol, atol=tol)

    def deriv(t, y):
        return float(np.asarray(fcn(t, y)).reshape(-1)[c])

    # PI steps until there is enough history
    times, solution, steps = [t0], [np.asarray(y0, dtype=float)], []
    while len(steps) < NUM_HIST + 3 and solver.t < tf:
        t_old = solver.t
        solver.step()
        steps.append(solver.t - t_old)
        times.append(solver.t)
        solution.append(solver.y.copy())
    derivs = [deriv(t, y) for t, y in zip(times, solution)]

    examples = []
    err_last = None
    while times[-1] < tf and len(steps) < MAX_STEPS:
        w = NUM_HIST + 1
        try:
            feature = build_feature(times[-w:], derivs[-w:],
                                    steps[-(NUM_HIST+2):], tol, solution[-1][c],
                                    stiffness_estimate(fcn, times[-1], solution[-1]))
        except (np.linalg.LinAlgError, ValueError):
            break  # derivative fit fails once the network's steps make the solution blow up
        if not np.all(np.isfinite(feature)):
            break
        # PI's response at this state: the step it would accept from here, starting from
        # its own proposal (at the first network decision) or from pi_factor of the error
        # of the network's last step, shrunk and retried while PI would reject it
        h_prop = solver.h_abs if err_last is None else pi_factor(err_last)*steps[-1]
        label = pi_accepted_step(solver, times[-1], solution[-1], h_prop) / steps[-1]
        examples.append((feature, label, len(steps), times[-1], solution[-1][c]))

        # Network chooses the next step
        x = (np.array(feature).reshape(1, -1) - train_mean) / train_std
        with torch.no_grad():
            r = float(np.exp(model(torch.tensor(x, dtype=torch.float32)).item()))
        h = min(steps[-1]*r, tf - times[-1])
        if not (np.isfinite(h) and h > 0):
            break

        y_new, err_last = trial_step(solver, times[-1], solution[-1], h)
        d_new = deriv(times[-1] + h, y_new)
        if not (np.all(np.isfinite(y_new)) and np.isfinite(err_last) and np.isfinite(d_new)):
            break
        times.append(times[-1] + h)
        solution.append(y_new)
        steps.append(h)
        derivs.append(d_new)

    if len(examples) > MAX_PER_TRAJECTORY:
        keep = rng.choice(len(examples), MAX_PER_TRAJECTORY, replace=False)
        examples = [examples[k] for k in sorted(keep)]
    return examples, c

def collect(model, train_mean, train_std, iteration):
    features, ratios, meta = [], [], []
    for name, f, params, y0s, span in JOBS:
        for ic in y0s:
            init = [ic] if np.isscalar(ic) else ic
            for tol in TOLS:
                with np.errstate(all="ignore"):
                    examples, c = rollout(f, span[0], init, span[1], tol, model, train_mean, train_std)
                for feature, label, i, t, y in examples:
                    features.append(feature)
                    ratios.append([label])
                    meta.append([name, str(ic), str(params), i, t, tol, c, y, label,
                                 f"dagger {iteration}"])
    return features, ratios, meta

if __name__ == "__main__":
    for iteration in range(1, N_ITERATIONS + 1):
        model, train_mean, train_std = load_model()
        features, ratios, meta = collect(model, train_mean, train_std, iteration)
        print(f"DAgger iteration {iteration}: {len(features)} new examples")

        for path, rows in [("features_fullmeta1.csv", features), ("ratios_fullmeta1.csv", ratios),
                           ("metadata_full1.csv", meta)]:
            with open(path, mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerows(rows)

        # Retrain every ensemble member on the enlarged dataset
        train_ensemble()
