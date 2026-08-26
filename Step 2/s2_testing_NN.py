import numpy as np
import math 
from scipy.integrate import RK45
from scipy.integrate._ivp.rk import rk_step
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from s2_rk45utils import *
import time

# Test trained NN 
#-----------------------------------------------------------------------
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12,64),
            nn.SiLU(), 
            nn.Linear(64,64),
            nn.SiLU(),
            nn.Linear(64,1)
        ) 
    
    def forward(self, x):
        output = self.net(x)
        return output

model = NN()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Define test ODES
def testode1(t,y):
    return -2*y  - 50*(y-math.cos(t)) - math.sin(t)
y0 = [0.75]
t_span = [0,5]
tol = 1e-8

def testode2(t,y): 
    return y*math.cos(t)

def testode3(t,y):
    return y*(1-y)

def testode4(t,y):
    return -y**3 

def testode5(t,y):
    return y**2

def testode_damped(t,Y):
    y,v= Y
    return [v, -0.2*v - y]
 
m = 1
v_tspan = [0, 4*math.pi]
y0v = [2.0,0.0]
def VanDerPol(t,Y):
    y,v = Y
    return [v, m*(1-y**2)*v - y]

def shm(t,Y):
    y,v = Y
    return [v,-y]

sigma = 10
r = 21
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

            # Create features to obtain error above and below desired tol
            feat_low = feature.copy()
            feat_low.append(0.5)

            feat_mid = feature.copy()
            feat_mid.append(1)

            feat_high = feature.copy()
            feat_high.append(1.25)

            # Predict next step with NN
            feat_high_ten = torch.tensor(feat_high, dtype=torch.float32)
            feat_mid_ten = torch.tensor(feat_mid, dtype=torch.float32)
            feat_low_ten = torch.tensor(feat_low, dtype=torch.float32)

            with torch.no_grad():
                high_err = model(feat_high_ten.unsqueeze(0)).item()
                mid_err = model(feat_mid_ten.unsqueeze(0)).item()
                low_err = model(feat_low_ten.unsqueeze(0)).item()
            
            # # Interpolate between values to find step - 2 points
            # trial_ratios = np.array([3/4, 3/2], dtype=float)
            # pred_errors = np.array([low_err, high_err], dtype=float)
            # target_tol = np.log(1)

            # Interpolate to find step - quadratic, 3 points
            trial_ratios = np.array([0.5, 1, 1.25], dtype=float)
            pred_errors = np.array([low_err, mid_err, high_err], dtype=float)
            target_tol = np.log(1)


            # Check that prediction behaves as expected before making r_new choice
            if not (low_err < mid_err < high_err):  # Predictions violate expected monotonicity.
                r_new = 1.0
                print("non-monotone")

            elif target_tol <= low_err:
                r_new = 0.5

            elif target_tol >= high_err:
                r_new = 1.5

            else:
                # r_new = 0.95*np.interp(target_tol, pred_errors, trial_ratios)

                quad_interp = interp1d(pred_errors, trial_ratios, kind='quadratic')
                r_new = 0.95*float(quad_interp(target_tol))

                h_trial = solver.h_abs*r_new

                # Compute trial error to see if rejected
                err_norm = compute_current_trial_err(solver, h_trial)
                print(f"NN prediction is {r_new} with error {err_norm}")

                max_retries = 15 # this is preventing accuracy
                retry = 0
                # old_r = r_new
                while err_norm > 1 and retry < max_retries:

                    old_r = r_new

                    x_ratios = [0.5, 1, r_new]
                    y_errors = [low_err, mid_err, np.log(err_norm)]

                    if np.min(y_errors) < target_tol < np.max(y_errors):

                        quad_interp2 = interp1d(y_errors, x_ratios, kind='quadratic')
                        candidate_r = 0.95*float(quad_interp2(target_tol))

                    else:
                        candidate_r = 0.95*old_r

                    r_new = candidate_r
                    h_trial = solver.h_abs*r_new
                    err_norm = compute_current_trial_err(solver, h_trial)

                    print(f"r has been adjusted to {candidate_r} with error {err_norm}")

                    retry += 1


                    # if err_norm <= 1 :
                    #     Valid_error = True
                    #     print(f"Final adjusted r_new is ", r_new)
                    # else:
                    #     print(f"r_new had been adjusted to ", r_new)

            # r_new = float(np.clip(r_new, 0.5, 1.5))
            solver.h_abs = solver.h_abs * r_new           

        nn_proposed_steps.append(solver.h_abs) 
        t_old = solver.t
        
        # Take RK step with NN prediction 
        solver.step()
        h = solver.t - t_old
        time_steps.append(float(h))

        # scale = solver.atol + solver.rtol * np.maximum(np.abs(solver.y_old), 
        #                                             np.abs(solver.y))
        
        # err = solver._estimate_error_norm(solver.K, h, scale)

        # errors.append(float(err))

    # Compute ratios
    ts_ratios = []
    for i in range(1, len(time_steps)):
        ratio = time_steps[i]/time_steps[i-1]
        ts_ratios.append(ratio)

    errors = solver.error_norms

    return (
        np.array(times), 
        np.array(solution), 
        np.array(time_steps),
        np.array(ts_ratios), 
        np.array(errors), 
        np.array(solver.rejected_error_norms),
        solver.rejected_steps,
        np.array(nn_proposed_steps),
        np.array(solver.rejections_per_step))

# Need to figure out how to adjust when error is very small*******
# def run_RK45_NN2(fcn, t0, y0, tf, tol, model, numhist, use_rejection=True):

    solver = RK45(fcn,t0, y0, tf, rtol=tol, atol=tol)
    model.eval()

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
    
            # Create features to obtain error above and below desired tol
            feat_low = feature.copy()
            feat_low.append(2/3)
    
            feat_high = feature.copy()
            feat_high.append(3/2)
    
            # Predict next step with NN
            feat_high_ten = torch.tensor(feat_high, dtype=torch.float32)
            feat_low_ten = torch.tensor(feat_low, dtype=torch.float32)
    
            with torch.no_grad():
                high_err = model(feat_high_ten.unsqueeze(0)).item()
                low_err = model(feat_low_ten.unsqueeze(0)).item()
    
            # Interpolate between values to find step
            trial_ratios = np.array([2/3, 3/2], dtype=float)
            pred_errors = np.array([low_err, high_err], dtype=float)
            target_tol = np.log(0.9) # bc error normalised

            # Check that prediction behaves as expected before making r_new choice
            if high_err <= low_err:  # Predictions violate expected monotonicity.
                r_new = 1.0

            elif target_tol <= low_err:
                r_new = 2/3

            elif target_tol >= high_err:
                r_new = 3/2

            else:
                r_new = np.interp(target_tol, pred_errors, trial_ratios)

            r_new = float(np.clip(r_new, 2/3, 3/2))
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


# Test performance of NN predictions
#--------------------------------------------------------------------------------

# First order ODE
start_time_pi = time.perf_counter()
times_pi, sol, ts_pi, _, errors_pi, rej_errors_pi, acc_steps_pi, rej_steps_pi, rej_per_step_pi = run_RK45(testode2,
                                                    t_span[0], y0, t_span[1], tol)
end_time_pi = time.perf_counter()

start_time_nn = time.perf_counter()
# times_nn, sol_nn, ts_nn, ts_rat, errors_nn, nn_steps, nn_rat, rej_err, acc_steps, rej_steps = run_RK45_NN2(testode4,
#                                         t_span[0], y0, t_span[1], tol, model, 4)
times_nn, sol_nn, ts_nn, _, errors_nn, rej_errors_nn, rej_steps_nn, nn_prop_steps, rej_per_step_nn = run_RK45_NN(testode2,
                                            t_span[0], y0, t_span[1], tol, model, 4)
end_time_nn = time.perf_counter()

# Diagnostics
print(f"Pi execution time: {end_time_pi-start_time_pi:.6f} seconds")
print(f"NN execution time: {end_time_nn-start_time_nn:.6f} seconds")

print(f"Pi rejected steps : ", rej_steps_pi)
print(f"Pi rejected errors : ", rej_errors_pi)

print(f"NN rejected steps : ", rej_steps_nn)
print(f"NN rejected errors : ", rej_errors_nn)

# print(rej_per_step_nn)


#-----------------------------
# Second Order ODE
# start_time_pi = time.perf_counter()
# times_pi, _, ts_pi, _, errors_pi, rej_errors_pi, _, rej_steps_pi, _ = run_RK45(shm, 
#                                                                 v_tspan[0], y0v, v_tspan[1], tol)
# end_time_pi = time.perf_counter()

# start_time_nn = time.perf_counter()
# times_nn, _, ts_nn, _, errors_nn, rej_errors_nn, rej_steps_nn, _, _ = run_RK45_NN(shm,
#                                                         v_tspan[0], y0v, v_tspan[1], tol, model, 4)
# end_time_nn = time.perf_counter()

# # Diagnostics
# print(f"Pi execution time: {end_time_pi-start_time_pi:.6f} seconds")
# print(f"NN execution time: {end_time_nn-start_time_nn:.6f} seconds")

# print(f"Pi rejected steps : ", rej_steps_pi)
# print(f"Pi rejected errors : ", rej_errors_pi)

# print(f"NN rejected steps : ", rej_steps_nn)
# print(f"NN rejected errors : ", rej_errors_nn)

# print(ts_pi)
# print(ts_nn)

#-------------------------------
# Lorenz/Rossler (3rd order system)
# start_time_pi = time.perf_counter()
# times_pi, sol, ts, errors_pi = run_RK45(Lorenz,L_t_span[0], L_y0, L_t_span[1], tol_L)
# end_time_pi = time.perf_counter()

# start_time_nn = time.perf_counter()
# times_nn, sol, ts, errors_nn, nn_steps, nn_rat, acc_steps, rej_steps = run_RK45_NN(Lorenz,L_t_span[0], L_y0, L_t_span[1], tol_L, model, 4)
# end_time_nn = time.perf_counter()

# print(f"Pi execution time: {end_time_pi-start_time_pi:.6f} seconds")
# print(f"NN execution time: {end_time_nn-start_time_nn:.6f} seconds")

#-----------------------------
# Measure Error - Compare the mappings of the controllers
N_pi = len(times_pi)
N_nn = len(times_nn)

print(f"Number of Pi steps : ", N_pi)
print(f"Number of NN steps : ", N_nn)

# Construct the normalized coordinates
pi_nvec = (1/(N_pi-1))*np.arange(0,N_pi)
nn_nvec = (1/(N_nn-1))*np.arange(0,N_nn)

# Plot distributions of timesteps
# plt.plot(pi_nvec, times_pi, label="pi")
# plt.plot(nn_nvec, times_nn, label = "nn")
# plt.legend(fontsize=15)
# plt.show()

# Compute interpolation of mappings over fine uniform grid
nxi = 5000
xi = np.linspace(0,1,nxi)

interp_pi = np.interp(xi, pi_nvec, times_pi)
interp_nn = np.interp(xi, nn_nvec, times_nn)

err_rms = np.sqrt(np.mean((interp_pi-interp_nn)**2))
err_rms_norm = err_rms/t_span[1] # across length of time interval
err_max = np.max(abs(np.diff(interp_pi-interp_nn)))
# print(f"RMS =", err_rms)
# print(f"Normalized RMS = ", err_rms_norm)
# print(f"Max discrepancy = ", err_max)

# Measure Error - difference to specified tol
tol_vec_pi = np.ones(len(errors_pi))
tol_vec_nn = np.ones(len(errors_nn))

err_pi = np.sqrt(np.mean((tol_vec_pi - errors_pi)**2))
err_nn = np.sqrt(np.mean((tol_vec_nn - errors_nn)**2))
print(f"RMS of diff between errors and tol for pi :", err_pi)
print(f"RMS of diff between errors and tol for nn :", err_nn)



