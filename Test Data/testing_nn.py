import numpy as np
import math 
from scipy.integrate import RK45
from scipy.integrate._ivp.rk import rk_step
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from rk45utils import *
from nn_model import load_ensemble
import time

# Import trained NN 
#-----------------------------------------------------------------------
# Average of the networks trained with each seed in nn_model.ENSEMBLE_SEEDS
model = load_ensemble()

# Load standardization variables
stats = np.load('standardization_NN3.npz')
train_mean = stats['mean']
train_std = stats['std']

# Test trained NN 
#-----------------------------------------------------------------------

# Define test ODES
# 1st Order
def testode1(t,y):
    return -2*y  - 50*(y-math.cos(t)) - math.sin(t)

def testode2(t,y): 
    return y*math.cos(t) -0.5*y
k=3
def testode3(t,y):
    return k*y*(1-y)

def testode4(t,y):
    return -0.5*y

def testode5(t,y):
    return y**2

# 2nd Order
def testode_damped(t,Y):
    y,v= Y
    return [v, -0.2*v - y]
 
m = 2
def VanDerPol(t,Y):
    y,v = Y
    return [v, m*(1-y**2)*v - y]

def shm(t,Y):
    y,v = Y
    return [v,-y]

# 3rd Order 
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


# Function to run RK45 and make NN predictions at each step
#------------------------------------------------------------------------------- 
def run_RK45_NN(fcn, t0, y0, tf, tol, model, numhist, sol_component=None):
    """Runs the RK45 solver and computes a NN at each step.
    PI controller outputs are not touched, we just compare the timestep choice of 
    each controller at the same point.
    sol_component - solution component used in features. Defaults to the one with
    the largest range, as in generate_data.
    """
    if sol_component is None:
        sol_component = most_varying_component(fcn, t0, y0, tf, tol)
    
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

    while solver.t < tf:

        solution.append(solver.y.copy()) 
        times.append(float(solver.t))

        prediction_made = False
        
        # Check if we have enough history 
        if len(time_steps) >= numhist + 3: 

            # Construct feature vector to pass to NN (same builder as generate_data)
            w = numhist + 1
            derivs = [float(np.asarray(fcn(t, y)).reshape(-1)[sol_component])
                      for t, y in zip(times[-w:], solution[-w:])]
            feature = build_feature(times[-w:], [y[sol_component] for y in solution[-w:]],
                                    derivs, time_steps[-(numhist+2):], tol,
                                    stiffness_estimate(fcn, times[-1], solution[-1]))

            # Standardize feature
            feature = np.array(feature).reshape(1, -1)
            feature = (feature - train_mean) / train_std

            # Make NN prediction
            feat_ten = torch.tensor(feature, dtype=torch.float32)
            with torch.no_grad():
                r_nn = float(np.exp(model(feat_ten).item()))  # network predicts log ratio

            h_nn = time_steps[-1]*r_nn
            
            prediction_made = True
        
        # Take RK step 
        t_old = solver.t
        solver.step()
        h_pi = solver.t - t_old

        if prediction_made:
            # Store nn predictions 
            nn_proposed_steps.append(h_nn) 
            nn_proposed_ratios.append(r_nn)

            # Compute pi ratios to compare with nn
            r_pi = h_pi/time_steps[-1]
            ratios.append(r_pi)   
        time_steps.append(float(h_pi))

        # scale = solver.atol + solver.rtol * np.maximum(np.abs(solver.y_old), 
        #                                             np.abs(solver.y))
        
        # err = solver._estimate_error_norm(solver.K, h, scale)

        # errors.append(float(err))
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


# Compare NN predictions to PI outputs
#--------------------------------------------------------------------------------

# Parameters (not for 3rd order)
y0 = [1.5]
# y0v = [2.0,0.0] 
t_span = [0,20]
# tols = [1e-4, 1e-5, 1e-6, 1e-8]
tols = [1e-6]

# Run testing - see if nn prediction replicate those of pi 
#----------------------------------------------------------------------------------------------------------
for t in tols:                                                  
    # Testing a 1st order ODE
    times_pi, sol_pi, ts_pi, ratios_pi_comp, errors_pi, _,_, ts_nn, ratios_nn, _ = run_RK45_NN(testode4,
                                                    t_span[0], y0, t_span[1], t, model, 4)
    
    # Metrics for evaluating 
    
    # RMSE between PI and NN ratios
    err_rms_rat = np.sqrt(np.mean((ratios_pi_comp[:-1]-ratios_nn[:-1])**2))
    print(f"Tol : {t}")
    print(f"RMSE : {err_rms_rat}")

    # Wrong Behaviour
    wrong_behav = []
    wrong_magnitudes = []
    under_pred = 0
    over_pred = 0
    for i in range(len(ratios_nn)):
        r1 = ratios_nn[i]
        r2 = ratios_pi_comp[i]
        if (r1-1)*(r2-1) < 0:
            wrong_behav.append(i)
            wrong_magnitudes.append(abs(r1-r2))
            if r1 < 1:
                under_pred += 1
            else:
                over_pred += 1
    wrong_magnitudes = np.array(wrong_magnitudes)
    wrong_behav_pct = len(wrong_magnitudes[wrong_magnitudes>0.2])/len(ratios_nn)
    print(f"Wrong behaviour % : {wrong_behav_pct}")

    # Relative Error 
    rel_err = abs((ratios_pi_comp[:-1]-ratios_nn[:-1]))/abs(ratios_pi_comp[:-1])
    av_rel_err = np.mean(rel_err)
    print(f"Average Rel Error : {av_rel_err}")
    print("--------------------------------------------------------------")

# Second Order ODE
# times_pi, sol, ts, errors, sh = run_RK45(VanDerPol,t_span[0], y0v, t_span[1], tol)
# times_nn, sol_nn, ts_nn, ratios_pi_comp, errors_nn, _,rej_steps, prop_steps, ratios_nn, _ = run_RK45_NN(VanDerPol,t_span[0], y0v, t_span[1], tol, model, 4)

# Plot the pi and nn ratios
plt.plot(ratios_pi_comp[:-1],label="PI ratio")
plt.plot(ratios_nn[:-1], label="NN predicted ratio")

plt.xlabel("Step")
plt.ylabel("Step-size ratio")
plt.legend()
plt.show()

# Second Order ODE
# times_pi, sol, ts, errors, sh = run_RK45(VanDerPol,t_span[0], y0v, t_span[1], tol)
# times_nn, sol_nn, ts_nn, ratios_pi_comp, errors_nn, _,rej_steps, prop_steps, ratios_nn, _ = run_RK45_NN(VanDerPol,t_span[0], y0v, t_span[1], tol, model, 4)

# Lorenz/Rossler
# times_pi, sol, ts, errors, sh = run_RK45(Lorenz,L_t_span[0], L_y0, L_t_span[1], tol_L)
# times_nn, sol, ts, errors, sh = run_RK45_NN(Lorenz,L_t_span[0], L_y0, L_t_span[1], tol_L, model, 4)

# NEEDS TO BE UPDATED FOR NEW FUNCTION
# # Measure Error - normalized time mappings of controllers
# #-------------------------------------------------------------
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


