import numpy as np
import math
from s2_rk45utils import *
from scipy.integrate import RK45
import matplotlib.pyplot as plt

# Exp Decay
lmda = 0.5
t_span = [0,4]
y0 = [0.5]
tol = 1e-9
def exp_decay(t,y):
    return -lmda*y

log_t_span = [0,5]
log_y0 = [0.125, 0.25, 0.5, 0.75, 2] 
def logistic(t,y):
    return y*math.cos(t)

times, sol, ts, errors, rej_errors, acc_steps, rej_steps, rej_ps = run_RK45(logistic,
                                            log_t_span[0], y0, log_t_span[1], tol)

print(f"rej steps : ", rej_steps)
print(f"rej per step : ", rej_ps)
print(f"rejected errors : ", rej_errors)
# print(len(times))
# solver = RK45(logistic, log_t_span[0], y0, log_t_span[1], rtol=tol, atol=tol)
# ratios = [0.5, 0.75, 1, 1.5, 2]
# index = 20

# trial_errors = []
# for r in ratios:
#     err, err_n = compute_trial_err(solver, index, r, times, sol, ts)
#     trial_errors.append(err_n)

# trial_errors = np.asarray(trial_errors)
# print(trial_errors)

# plt.figure(figsize=(7, 5))

# plt.plot(
#     ratios,
#     np.log10(trial_errors),
#     linewidth=2
# )

# plt.axhline(0,
#     linestyle="--",
#     label=r"Acceptance boundary: $\log_{10}(E)=0$"
# )

# plt.axvline(
#     1,
#     linestyle=":",
#     label=r"Same step size: $h_{\mathrm{trial}}/h=1$"
# )

# plt.xlabel(r"Trial step-size ratio $h_{\mathrm{trial}}/h$")
# plt.ylabel(r"$\log_{10}(\mathrm{normalized\ error})$")
# plt.title(f"Trial Step-Size Ratio vs. Error at Step {index}")

# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()

# print(f"err = ", err)
# print(f"err norm = ", float(err_n))


# x = np.array([2, 3])
# y = np.array([4, 2])
# # x_new = np.linspace(2/3, 3/2, 5)
# target = 3
# order = np.argsort(y)
# y_new = np.interp(target, y[order], x[order])
# print(y_new)

# v=np.ones(6)*5
# print(v)
