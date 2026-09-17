import numpy as np
import math 
from scipy.integrate import RK45
from scipy.integrate._ivp.rk import rk_step
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rk45utils import *

# Import trained NN 
#-----------------------------------------------------------------------
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11,48),
            nn.Tanh(), 
            nn.Linear(48,48),
            nn.Tanh(),
            nn.Linear(48,1)
        ) 
    
    def forward(self, x):
        output = self.net(x)
        return output

model = NN()
model.load_state_dict(torch.load('model_weights_s1.pth'))
model.eval()

def take_RK45_step(solver, fcn, times, solution, tf, tol, h):

    # solver = RK45_counting(fcn,t0, y0, tf, rtol=tol, atol=tol)
    t = times[-1]
    y = solution[-1]
    # h_trial = min(h_trial, solver.t_bound - t)

    K_trial = np.empty((solver.n_stages + 1, solver.n), dtype=solver.y.dtype)
    fcn = solver.fun
    f = np.asarray(fcn(t,y), dtype=float)

    y_new, _ = rk_step(fcn, t, y, f, h, solver.A, solver.B, 
                                   solver.C, K_trial)
    return y_new

def run_solverNN(fcn, t0, y0, tf, tol, model, numhist):

    solver = RK45_counting(fcn,t0, y0, tf, rtol=tol, atol=tol)
    model.eval()

    # Storage arrays
    times = []
    solution = []
    time_steps = []
    ratios = []
    # errors = []
    # nn_proposed_steps = []
    # nn_proposed_ratios = []
    t = t0
    y = y0
    times.append(t)
    solution.append(y)

    while t < tf:

        if len(times) <= numhist + 3:
            print("1")
            t_old = solver.t
            solver.step()
            h_pi = solver.t - t_old

            time_steps.append(h_pi)
            times.append(float(solver.t))
            solution.append(solver.y.copy())

            t = solver.t

        else:
            print("2")
            # Construct feature vector to pass to NN
            feature = []

            for i in range(numhist,-1,-1):
                feature.append(float(solution[-1-i][0]))  
            
            for j in range(numhist, -1,-1):
                ratio = time_steps[-1-j]/time_steps[-2-j]
                feature.append(ratio)
            
            feature.append(float(np.log(tol)))

            # Predict next step with NN
            feat_ten = torch.tensor(feature, dtype=torch.float32)
            with torch.no_grad():
                r_nn = model(feat_ten.unsqueeze(0)).item()

            h_nn = time_steps[-1]*r_nn

            # Check that we do not surpass tf
            if t + h_nn > tf:
                h_nn = tf - t

            y_next = take_RK45_step(solver, fcn, times, solution, tf, tol, h_nn)
            t = times[-1] + h_nn

            time_steps.append(h_nn)
            times.append(t)
            solution.append(y_next)

    return (
        np.array(times), 
        np.array(solution), 
        np.array(time_steps), 
        np.array(ratios)
    )

def testode(t,y):
    return -0.5*y + y*(1-y)

def testode2(t,y):
    return y*math.cos(t)
y0 = [1.1]
t_span = [0,5]
tol = 1e-3

# times_nn, sol_nn, ts_nn, rat_nn = run_solverNN(testode, t_span[0], y0, t_span[1], tol, model, 4)
# times, sol, ts, rat, _,_,_,_,_ = run_RK45(testode, t_span[0], y0, t_span[1], tol)

# print(f"Number of Steps PI:", len(ts))
# print(f"Number of Steps NN:", len(ts_nn))
# plt.plot(sol, label = 'PI')
# plt.plot(sol_nn, label = 'NN')
# plt.xlabel("# of steps")
# plt.legend()
# plt.show()


# Compute Error metric
#----------------------------------------------
# Find 'exact' solution with PI
exact_tol = 1e-13

times_ex, sol_ex, ts_ex,_,_,_,_,_,_ = run_RK45(testode, t_span[0], y0, t_span[1], exact_tol)
# print(f"exact steps : ", len(ts_ex))

# Run NN and PI for looser tol
test_tol = 1e-5
times_nn, sol_nn, ts_nn, rat_nn = run_solverNN(testode, t_span[0], y0, t_span[1], test_tol, model, 4)
times, sol, ts, rat, _,_,_,_,_ = run_RK45(testode, t_span[0], y0, t_span[1], test_tol)

print(f" ts steps pi :", ts)
print(f"ts steps nn :", ts_nn)

# Compute error from exact
# do interpolation of mappings to compare 
N_ex = len(times_ex)
N_pi = len(times)
N_nn = len(times_nn)

print(f"Exact steps: ", N_ex)
print(f"Pi steps: ", N_pi)
print(f"NN steps: ", N_nn)

# Construct the normalized coordinates
ex_nvec = (1/(N_ex-1))*np.arange(0,N_ex)
pi_nvec = (1/(N_pi-1))*np.arange(0,N_pi)
nn_nvec = (1/(N_nn-1))*np.arange(0,N_nn)

# Plot distributions of timesteps
plt.figure(1)
plt.plot(ex_nvec, times_ex, label="exact")
plt.plot(pi_nvec, times, label="pi")
plt.plot(nn_nvec, times_nn, label = "nn")
plt.legend(fontsize=15)
plt.show()

# Compute interpolation of mappings over fine uniform grid
nxi = 5000
xi = np.linspace(0,1,nxi)

interp_ex = np.interp(xi, ex_nvec, times_ex)
interp_pi = np.interp(xi, pi_nvec, times)
interp_nn = np.interp(xi, nn_nvec, times_nn)

err_rms_pi = np.sqrt(np.mean((interp_pi-interp_ex)**2))
err_rms_nn = np.sqrt(np.mean((interp_nn-interp_ex)**2))
# err_rms_norm = err_rms/t_span[1] # across length of time interval
# err_max = np.max(abs(np.diff(interp_pi-interp_nn)))
print(f"RMS PI =", err_rms_pi)
print(f"RMS NN =", err_rms_nn)
# print(f"Normalized RMS = ", err_rms_norm)
# print(f"Max discrepancy = ", err_max)
err_pi = abs(interp_pi-interp_ex)
err_nn = abs(interp_nn-interp_ex)

# plt.figure(2)
# plt.plot(err_pi, label="pi")
# plt.plot(err_nn, label="nn")
# plt.legend()
# plt.show()



