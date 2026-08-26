import numpy as np
import math
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import csv
from s2_rk45utils import *

# ODEs used to generate data
# ------------------------------------------------------------------
# Exponential Decay
lmda = 2  # increase to get more points
ed_t_span = [0,5]
ed_y0 = [1] # increase to get more points
def exp_decay(t,y):
    return -lmda*y

# Logistic
log_t_span = [0,5]
log_y0 = [0.125, 0.25, 0.5, 0.75, 2] 
def logistic(t,y):
    return y*(1-y)

# Multiple Scales
ms_t_span = [0, 2*math.pi]
ms_y0 = [-2, -1, 0, 0.5, 1, 2]
def multi_scales(t,y):
    return -1000*(y-math.cos(t)) - math.sin(t)

# Oscillatory
o_t_span = [0, 4*math.pi]
o_y0 = [-2, -1, -0.5, 0.5, 1, 2]
def oscil(t,y):
    return y*math.cos(t)

# Simple Harmonic Motion
shm_t_span = [0, 4*math.pi]
shm_y0 = [[0.5, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [3.0, -1.0],]
def shm(t,Y):
    y,v = Y
    return [v,-y]

# VanDerPol
m = [0.5, 1, 2]
vdp_t_span = [0, 4*math.pi]
vdp_y0 = [[0.1, 0.0], [2.0, 0.0], [0.0, 2.0], [1.0, -1.0], [3.0, 0.0]]

VDP_fcns = []
for val in m:
    def VanDerPol(t,Y):
        y,v = Y
        return [v, val*(1-y**2)*v - y]
    VDP_fcns.append(VanDerPol)

# Damped Harmonic Motion
d_t_span = [0, 6*math.pi]
d_y0 = [[0.5, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [3.0, -1.0],]
def damped_hm(t,Y):
    y,v= Y
    return [v, -0.2*v - y]

# ODE that blows up
bu_t_span = [0,0.95]
bu_y0 = [0.75, 0.85,0.95, 1]
def blows_up(t,y):
    return y**2

# Lorenz
sigma = 10
r_vals = [28, 317]
b = 8/3
L_t_span = [0,5]
L_y0 = [[0.0, 1.0, 20.0], [1.0, 1.0, 1.0]]
L_fcns = []
for r in r_vals:
    def Lorenz(t,Y):
        x,y,z = Y
        return [sigma*(y-x), r*x-y-x*z, x*y-b*z]
    L_fcns.append(Lorenz)

# Functions for Data Generating
#---------------------------------------------------------------------

# Function to choose data points uniformly for exp_decay
def select_points_exp(sol):
    """Given a skeleton of a solution, select every 6th point for the dataset. 
    Return the indices of selected points"""

    indices = []
    n = len(sol)
    index = math.floor(n/6) 
    j = 1
    while j*index < n:
        indices.append(j*index)
        j += 1

    return indices

def build_features_trial_err(solver, indices, full_solution, sol_comp, times, time_steps, 
                             next_step_pct, num_hist, tol):
    """ Builds a feature vector containing solution and ratio history, log(tol), and trial ratio.
    Feature vector is of length: (num_hist + 1)*2 +2
    I.e. : num_hist = 2, index i
    => vec = [sol[i-2], sol[i-1], sol[i], r[i-2], r[i-1], r[i], log(tol), trial_ratio]
    indices - of chosen data points to be added to dataset
    num_hist - number of previous solution points (adds numhist+1 items to feature)
    """
    data_list = []
    trial_step_errors = []
    for i in indices:

        # Check if index is at the end of the time span
        if i == len(sol_comp)-1:
            pass
        # Check if not enough solution history
        elif i < num_hist:
            print("not enough solution history at this point")
        else:   
            for n in next_step_pct: 

                feature = []
                for k in range(num_hist, -1, -1):
                    feature.append(sol_comp[i-k])

                for j in range(num_hist, -1, -1):
                    r = time_steps[i-1-j]/time_steps[i-2-j]
                    feature.append(r)

                feature.append(np.log(tol))

                h = time_steps[i-1] 
                h_next = n*h
                feature.append(h_next/h)

                data_list.append(feature)

                # Create vector of trial time step errors
                err, err_norm = compute_trial_err(solver, i, n, times, full_solution, time_steps)
                trial_step_errors.append(float(np.log(err_norm)))

    return data_list, trial_step_errors

# Function to generate data matrix and target ratios
def generate_data(fcn, t0, y0, tf, tol, ord):
    """Generate a dataset and target error vector from the passed parameters. 
    - Higher order ODEs must be written as a system of 1st order ODEs. The solution information 
    from the component that varies the most is used in dataset.

    Parameters:
    fcn - RHS of ODE (or of system of 1st order ODEs)  
    time span: [t0,tf]
    y0 - list of initial values => i.e. y0 = [a,....,b] or y0 = [[a,b],...,[c,d]] for 2nd order
    tol - list of tolerance
    
    Function loops through y0 and tol, generating a feature vector for each combination of 
    values in lists. 
    For each value in y0/tol, a step is taken with run_RK45. Then certain examples are
    selected for the dataset. The selected indices are passed to build_features_trial_err."""

    Data_set = []
    Errors = []

    for i in y0:
        for j in tol:
            if ord ==1:
                init_value = [i]
                # times, sol, ts, err, sh = run_RK45(fcn, t0, [i], tf, j)
            else:
                init_value = i
                # times, sol, ts, err, sh = run_RK45(fcn, t0, i, tf, j)
            # check that thee outputs are correct
            times, sol, ts, err, rej_err, acc_steps, rej_steps, rejections_per_step = run_RK45(fcn, 
                                                    t0, init_value, tf, j) 
            
            sol_comps = []
            for k in range(0,len(sol[0])):
                sol_comps.append(np.array(sol[:,k]))

            diffs = np.ptp(sol_comps, axis=1)
            max_var = np.argmax(diffs)

            full_sol = sol
            sol = sol[:,max_var]

            # sol = sol[:,0] # change this line
             
            yp = approx_first_deriv2(sol, times)
            yp2 = approx_second_deriv2(sol, times)

            # Select desired points
            if fcn == exp_decay:
                indices = select_points_exp(sol)
            
            else:
                I = []

                # Find sections with largest derivs
                max_indices_yp = np.where(yp == yp.max())[0]
                max_indices_yp2 = np.where(yp2 == yp2.max())[0]

                # Find sections with smallest derivs
                min_indices_yp = np.where(yp == yp.min())[0]
                min_indices_yp2 = np.where(yp2 == yp2.min())[0]

                # Find sections with ~0 derivative
                zero_deriv_yp = np.where(abs(yp) < 1e-4)[0]
                zero_deriv_yp2 = np.where(abs(yp2) < 1e-4)[0]

                window = 2  # size of interval 
                for idx in np.concatenate((max_indices_yp, max_indices_yp2, min_indices_yp,
                                           min_indices_yp2,zero_deriv_yp, zero_deriv_yp2)):
                    start = max(6, idx-window) # 6 = num_hist + window
                    
                    end = min(len(sol), idx+window+1)

                    I.extend(range(start, end))

                I.append(len(sol)-1) # Add last data point to find differences 
                I = np.unique(I) # Get rid of duplicates

                if len(I) > 2 :
                    diffs = np.diff(I)
                    order = np.argsort(diffs)
                    
                    # Compute "boring" indices
                    new_indices = []
                    
                    for k in [0,1]:
                        Gap_index = order[-1-k]

                        mid = I[Gap_index] + math.floor(0.5*(I[Gap_index+1]-I[Gap_index]))
                        new_indices.append(int(mid))

                    # Add new indices to I
                    I = np.append(I, new_indices)

                indices = np.unique(I)

            next_steps = [0.75, 0.9, 1, 1.1, 1.25, 1.5]
            solver = RK45(fcn, t0, init_value, tf, rtol=j, atol=j) # can we get rid of somehow?
            data, errors = build_features_trial_err(solver, indices, full_sol, sol, times, ts, next_steps, 4, j)

            Data_set.extend(data)
            Errors.extend(errors)

    
    return Data_set, Errors

# Construct Data Matrix and Target Ratios
#------------------------------------------------------------------------

# Generate data from above ODEs (first and second order)
Data_set = []
Errors = []

# 1. Exp Decay
Data_exp, Sr_exp = generate_data(exp_decay, ed_t_span[0], ed_y0, ed_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10, 1e-12],1)
print("Examples generated with Exp Decay: ", len(Data_exp))
Data_set.extend(Data_exp)
Errors.extend(Sr_exp)
print("Examples generated with Exp Decay: ", len(Data_exp))
print("Total examples: ", len(Data_set))

#2. Logistic
Data_log, Sr_log = generate_data(logistic, log_t_span[0], log_y0, log_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10, 1e-12],1)
Data_set.extend(Data_log)
Errors.extend(Sr_log)
print("Examples generated with Logistic: ", len(Data_log))
print("Total examples: ", len(Data_set))

#3. Multiple Scales 
Data_ms, Sr_ms = generate_data(multi_scales, ms_t_span[0], ms_y0, ms_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10, 1e-12],1)
Data_set.extend(Data_ms)
Errors.extend(Sr_ms)
print("Examples generated with Multiple Scales: ", len(Data_ms))
print("Total examples: ", len(Data_set))

#4. Oscillating
Data_o, Sr_o = generate_data(oscil, o_t_span[0], o_y0, o_t_span[1], [1e-6, 1e-8, 1e-10, 1e-12],1)
Data_set.extend(Data_o)
Errors.extend(Sr_o)
print("Examples generated with Oscillating: ", len(Data_o))
print("Total examples: ", len(Data_set))

#5. Simple Harmonic Motion 
Data_shm, Sr_shm = generate_data(shm, shm_t_span[0], shm_y0, shm_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10], 2)
Data_set.extend(Data_shm)
Errors.extend(Sr_shm)
print("Examples generated with SHM : ", len(Data_shm))
print("Total examples: ", len(Data_set))

#6. VanDerPol
Total_ex = 0
for f in VDP_fcns:
    Data_vdp, Sr_vdp = generate_data(f, vdp_t_span[0], vdp_y0, vdp_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10], 2)
    Data_set.extend(Data_vdp)
    Errors.extend(Sr_vdp)
    Total_ex += len(Data_vdp)
print("Examples generated with VanDerPol: ", Total_ex)
print("Total examples: ", len(Data_set))

#7. Damped Harmonic Motion
Data_d, Sr_d = generate_data(damped_hm, d_t_span[0], d_y0, d_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10], 2)
Data_set.extend(Data_d)
Errors.extend(Sr_d)
print("Examples generated with Damped HM : ", len(Data_d))
print("Total examples: ", len(Data_set))

#8. ODE that Blows up 
Data_bu, Sr_bu = generate_data(blows_up, bu_t_span[0], bu_y0, bu_t_span[1], [1e-6, 1e-8, 1e-10, 1e-12],1)
Data_set.extend(Data_bu)
Errors.extend(Sr_bu)
print("Examples generated with y' = y**2: ", len(Data_bu))
print("Total examples: ", len(Data_set))

# #9. Lorenz - standard parameters
# Data_l, Sr_l = generate_data(Lorenz, L_t_span[0], L_y0, L_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10],3)
# Data_set.extend(Data_l)
# Ratios.extend(Sr_l)
# print("Examples generated with Lorenz: ", len(Data_l))
# print("Total examples: ", len(Data_set))

#9. Lorenz
Total_ex = 0
for f in L_fcns:
    Data_L, Sr_L = generate_data(f, L_t_span[0], L_y0, L_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10], 3)
    Data_set.extend(Data_L)
    Errors.extend(Sr_L)
    Total_ex += len(Data_L)
print("Examples generated with Lorenz: ", Total_ex)
print("Total examples: ", len(Data_set))


print(f"Size of Data Matrix: {len(Data_set)}x{len(Data_set[0])}" )

# Store Dataset as a csv file
#----------------------------------------------------------------------------
with open("features.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(Data_set)

with open("errors.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows([[r] for r in Errors])

    










