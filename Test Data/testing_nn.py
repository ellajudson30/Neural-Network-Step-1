import numpy as np
import math 
from scipy.integrate import RK45
from scipy.integrate._ivp.rk import rk_step
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rk45utils import *
import time

# Test trained NN 
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

# Test trained NN 
#-----------------------------------------------------------------------

# Define test ODES
def testode1(t,y):
    return -2*y  - 50*(y-math.cos(t)) - math.sin(t)
y0 = [1]
t_span = [0,5]
tol = 1e-10

def testode2(t,y): 
    return y*math.cos(t)

def testode3(t,y):
    return y*(1-y)

def testode4(t,y):
    return -y

def testode5(t,y):
    return y**2

def testode_damped(t,Y):
    y,v= Y
    return [v, -0.2*v - y]
 
m = 3
y0v = [2.0,0.0]
def VanDerPol(t,Y):
    y,v = Y
    return [v, m*(1-y**2)*v - y]

def shm(t,Y):
    y,v = Y
    return [v,-y]

sigma = 10
r = 28
b = 8/3
L_t_span = [0,5]
tol_L = 1e-9
L_y0 = [0.0, 1.0, 20.0] # classic IC for standard params
def Lorenz(t,Y):
    x,y,z = Y
    return [sigma*(y-x), r*x-y-x*z, x*y-b*z]

# Rossler System
A = 0.2
B = 0.2
C = 5.7
r_t_span = [20,40]
r_y0 = [0,0,0]
def Rossler(t,Y):
    x,y,z = Y
    return [-y-z, x + A*y, B+z*(x-C)]


# Function to run RK45 with NN choosing the next step
#------------------------------------------------------------------------------- 
def run_RK45_NN(fcn, t0, y0, tf, tol, model, numhist):
    """Takes an RK45 step and uses the prediction from the trained NN as the next step.
    """
    
    solver = RK45_counting(fcn,t0, y0, tf, rtol=tol, atol=tol)
    model.eval()

    # Storage arrays
    times = []
    solution = []
    time_steps = []
    ratios = []
    errors = []
    nn_proposed_steps = []
    nn_proposed_ratios = []
    # nn_proposed_times = []
    # nn_proposed_times.append(t0)

    while solver.t < tf:

        solution.append(solver.y.copy()) 
        times.append(float(solver.t))

        prediction_made = False
        
        # Check if we have enough history 
        if len(time_steps) > numhist + 3: 

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
                # print(r_new)

            # # Prevent pathological NN predictions
            # r_new = np.clip(r_new, 0.2, 3.0)
            # print(r_new)

            h_nn = time_steps[-1]*r_nn
            
            # Alter pi controller
            # solver.h_abs = solver.h_abs * r_new


            prediction_made = True
        
        # Take RK step 
        t_old = solver.t
        solver.step()
        h_pi = solver.t - t_old

        if prediction_made:
            nn_proposed_steps.append(h_nn) 
            nn_proposed_ratios.append(r_nn)

            r_pi = h_pi/time_steps[-1]
            ratios.append(r_pi)   
        time_steps.append(float(h_pi))

        # scale = solver.atol + solver.rtol * np.maximum(np.abs(solver.y_old), 
        #                                             np.abs(solver.y))
        
        # err = solver._estimate_error_norm(solver.K, h, scale)

        # errors.append(float(err))
    
    # fix how this is computed to match when predictions are being made
    # Compute ratios
    # pi_ratios = []
    # for i in range(1, len(time_steps)):
    #     ratio = time_steps[i]/time_steps[i-1]
    #     pi_ratios.append(ratio)

    errors = solver.error_norms

    return (
        np.array(times), 
        np.array(solution), 
        np.array(time_steps), 
        np.array(ratios),
        np.array(errors), 
        np.array(solver.rejected_error_norms),
        solver.rejected_steps,
        np.array(nn_proposed_steps),
        np.array(nn_proposed_ratios),
        np.array(solver.rejections_per_step))

# def run_RK45_NN2(fcn, t0, y0, tf, tol, model, numhist, use_rejection=True):

    solver = RK45(fcn,t0, y0, tf, rtol=tol, atol=tol)
    model.eval()

    # max_steps = 100_000
    # min_h = 1e-14 * max(1.0, abs(t), abs(tf))   

    # Current integration state - for rk_step
    t = float(solver.t)
    y = solver.y.copy()
    f = solver.f.copy()

    # Storage arrays and counters
    times = [t]
    solution = [y.copy()]
    time_steps = []
    errors = []
    nn_proposed_steps = []
    nn_proposed_ratios = []
    accepted_steps = 0
    rejected_steps = 0
    rejected_errors = []

    # Start with SciPy's initial-step estimate
    h_abs = float(solver.h_abs)
    
    while t < tf:

        r_new = 1 

        # Check if we have enough history 
        if len(time_steps) > numhist+1: 
    
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
                model.eval()
                r_new = model(feat_ten.unsqueeze(0)).item()

            if not np.isfinite(r_new):
                raise RuntimeError(
                    f"NN predicted a non-finite ratio at t={t}: {r_new}"
                )

            # Prevent zero, negative, or extreme predictions.
            r_new = float(np.clip(r_new, 0.2, 3))
            h_abs = h_abs * r_new
            
        nn_proposed_ratios.append(r_new)

        # Make sure to not step beyond final time
        h_abs = min(h_abs, tf - t)
        nn_proposed_steps.append(h_abs)

        # Use NN to replace role of Controller
        valid_step = False
        while not valid_step:

            # Take an RK Step
            h = h_abs
            y_new, f_new = rk_step(fcn, t, y, f, h, solver.A, solver.B, 
                                solver.C, solver.K)
            
            # Compute error norm estimation
            scale = solver.atol + solver.rtol * np.maximum(np.abs(y), 
                                                            np.abs(y_new))
                
            err_norm = solver._estimate_error_norm(solver.K, h, scale)

            if not np.isfinite(err_norm):
                raise RuntimeError(
                    f"Non-finite error estimate at t={t}, h={h}: "
                    f"{err_norm}"
                )
            
            # Accept or Reject Step
            if not use_rejection or err_norm <= 1.0:
                valid_step = True
            
            else:
                rejected_steps += 1
                h_abs = h_abs * 2/3
                rejected_errors.append(err_norm)

        # Accept the successful step
        t += h
        y = y_new
        f = f_new

        accepted_steps += 1

        times.append(t)
        solution.append(y.copy())
        time_steps.append(h)
        errors.append(float(err_norm))

        print(f"t={t:.6f}, h={h:.3e}, "
              f"ratio={r_new:.3f}, err={err_norm:.3e}"
    )
    
    return (
            np.array(times), 
            np.array(solution), 
            np.array(time_steps), 
            np.array(errors), 
            np.array(nn_proposed_steps),
            np.array(nn_proposed_ratios),
            np.array(rejected_errors),
            accepted_steps,
            rejected_steps )

# # Make fcn that uses NN ratios but not solver.step() - use rk_step()
# def run_RK45_justNN(fcn, t0, y0, tf, tol, model, numhist):
    solver = RK45(fcn,t0, y0, tf, rtol=tol, atol=tol)
    model.eval()

    # Storage arrays
    times = []
    solution = []
    time_steps = []
    errors = []
    nn_proposed_steps = []

    while solver.t < tf:

        solution.append(solver.y.copy()) 
        times.append(float(solver.t))
        
        # Check if we have enough history 
        if len(time_steps) > numhist+2: 

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
                r_new = model(feat_ten.unsqueeze(0)).item()
    return 

# Test performance of NN predictions
#--------------------------------------------------------------------------------

# First order ODE
times_pi, sol, ts, ratios_pi, errors, _, _, _, _ = run_RK45(testode2,t_span[0], y0, t_span[1], tol)
# times_nn, sol_nn, ts_nn, errors_nn, prop_steps_nn, prop_ratios, rejected_errors_nn, acc_steps, rej_steps = run_RK45_NN(testode3,
#                                                                     t_span[0], y0, t_span[1], tol, model, 4)

times_nn, sol_nn, ts_nn, ratios_pi_comp, errors_nn, _,rej_steps, prop_steps, ratios_nn, _ = run_RK45_NN(testode2,t_span[0], y0, t_span[1], tol, model, 4)

plt.plot(errors)
plt.show()

print(len(ratios_nn))
print(len(ratios_pi_comp))
print(ratios_nn-ratios_pi_comp)
err_rms_rat = np.sqrt(np.mean((ratios_pi_comp[:-1]-ratios_nn[:-1])**2))
print(err_rms_rat)

plt.plot(ratios_pi_comp, label="PI ratio")
plt.plot(ratios_nn, label="NN predicted ratio")

plt.xlabel("Step (with predictions)")
plt.ylabel("Step-size ratio")
plt.legend()
plt.show()


# Second Order ODE
# times_pi, sol, ts, errors, sh = run_RK45(VanDerPol,t_span[0], y0v, t_span[1], tol)
# times_nn, sol, ts, errors, sh = run_RK45_NN(VanDerPol,t_span[0], y0v, t_span[1], tol, model, 4)

# Lorenz/Rossler
# times_pi, sol, ts, errors, sh = run_RK45(Lorenz,L_t_span[0], L_y0, L_t_span[1], tol_L)
# times_nn, sol, ts, errors, sh = run_RK45_NN(Lorenz,L_t_span[0], L_y0, L_t_span[1], tol_L, model, 4)

# Measure error - rmse of predicted time steps (not very comparable) 
# predicted = np.array(ts_nn[5:100])
# actual = np.array(ts[5:100])
# rmse = np.sqrt(np.mean((predicted-actual)**2))
# rmse_steps = np.sqrt(np.mean((predicted-np.array(nn_steps)[5:100])**2))
# print(rmse)
# print(rmse_steps)

# # Measure Error - Compare the mappings of the controllers
# N_pi = len(times_pi)
# N_nn = len(times_nn)

# print(f"Pi steps: ", N_pi)
# print(f"NN steps: ", N_nn)

# # Construct the normalized coordinates
# pi_nvec = (1/(N_pi-1))*np.arange(0,N_pi)
# nn_nvec = (1/(N_nn-1))*np.arange(0,N_nn)

# # Plot distributions of timesteps
# plt.plot(pi_nvec, times_pi, label="pi")
# plt.plot(nn_nvec, times_nn, label = "nn")
# plt.legend(fontsize=15)
# plt.show()

# # Compute interpolation of mappings over fine uniform grid
# nxi = 5000
# xi = np.linspace(0,1,nxi)

# interp_pi = np.interp(xi, pi_nvec, times_pi)
# interp_nn = np.interp(xi, nn_nvec, times_nn)

# err_rms = np.sqrt(np.mean((interp_pi-interp_nn)**2))
# err_rms_norm = err_rms/t_span[1] # across length of time interval
# err_max = np.max(abs(np.diff(interp_pi-interp_nn)))
# print(f"RMS =", err_rms)
# print(f"Normalized RMS = ", err_rms_norm)
# print(f"Max discrepancy = ", err_max)

# # Try to split time span into section and compute error over each section

