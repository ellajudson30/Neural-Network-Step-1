import numpy as np
import math
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import csv
from rk45utils import *

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

def build_features(indices, solution, time_steps, stage_history):
    """For each example indicated in indices, builds a feature vector of length 8. 
    I.e. : At example of index i, vector contains (in order):
    current solution value - solution[i]
    stage history at i - Stage_history[i-1][k], k=0,1,...,5   (n_stages=6 for RK45)
    current time step ratio at time[i] - timestep[i]/timestep[i-1]

    Then builds a vector of target ratios: timestep[i+1]/timestep[i]
    """
    data_list = []
    next_step_ratio = []

    for i in indices:
        feature = []
        feature.append(float(solution[i]))

        for j in range(0, RK45.n_stages): 
            feature.append(float(stage_history[i-1][j][0]))
    
        r = time_steps[i]/time_steps[i-1]
        feature.append(r)

        data_list.append(feature)

    # Create vector of time step ratios
    for i in indices:
        r_next = time_steps[i]/time_steps[i-1]
        next_step_ratio.append(r_next)

    return data_list, next_step_ratio
def build_features_solhist(indices, solution, time_steps, num_hist, tol):
    """ Builds a feature vector containing solution and ratio history and log(tol).
    Feature vector is of length: (num_hist + 1)*2 +1
    I.e. : num_hist = 2, index i
    => vec = [sol[i-2], sol[i-1], sol[i], r[i-2], r[i-1], r[i], log(tol)]
    indices - of chosen data points to be added to dataset
    num_hist - number of previous solution points (adds numhist+1 items)
    """
    data_list = []
    next_step_ratio = []
    for i in indices:

        # Check if index is at the end of the time span
        if i == len(solution)-1:
            pass
        # Check if not enough solution history
        elif i < num_hist:
            print("not enough solution history at this point")
        else:
            feature = []
            for k in range(num_hist, -1, -1):
                feature.append(solution[i-k])

            for j in range(num_hist, -1, -1):
                r = time_steps[i-j]/time_steps[i-j-1]
                feature.append(r)

            feature.append(np.log(tol))

            data_list.append(feature)

            # Create vector of next time steps
            r_next = time_steps[i+1]/time_steps[i]
            next_step_ratio.append(r_next)


    return data_list, next_step_ratio

# Function to generate data matrix and target ratios
def generate_data(fcn, t0, y0, tf, tol, ord):
    """Generate a dataset and target ratio vector from the passed parameters.
    - Works for 1st and 2nd order ODEs. 
    - 2nd order must be written as a system of 1st order ODEs. Only the solution information 
    from y(t) is used is dataset.

    Parameters:
    fcn - RHS of ODE (or of system of 1st order ODEs)  
    time span: [t0,tf]
    y0 - list of initial values => y0 = [a,....,b] or y0 = [[a,b],...,[c,d]] for 2nd order
    tol - list of tolerance
    
    Function loops through y0 and tol, generating a feature vector for each combination of 
    values in lists. 
    For each value in y0/tol, a step is taken with run_RK45. Then certain examples are
    selected for the dataset. The selected indices are passed to build_features_solhist."""

    Data_set = []
    Step_ratios = []

    for i in y0:
        for j in tol:
            if ord ==1:
                times, sol, ts, err, sh = run_RK45(fcn, t0, [i], tf, j)
            elif ord == 2:
                times, sol, ts, err, sh = run_RK45(fcn, t0, i, tf, j)
            
            sol = sol[:,0]
             
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

            # Create features/ratios and add to Dataset
            data, ratios = build_features_solhist(indices, sol, ts, 4, j)
            # data, ratios = build_features(indices, sol, ts, sh)

            Data_set.extend(data)
            Step_ratios.extend(ratios)

    
    return Data_set, Step_ratios

# Construct Data Matrix and Target Ratios
#------------------------------------------------------------------------

# Generate data from above ODEs (first and second order)
Data_set = []
Ratios = []

# 1. Exp Decay
Data_exp, Sr_exp = generate_data(exp_decay, ed_t_span[0], ed_y0, ed_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10, 1e-12],1)
print("Examples generated with Exp Decay: ", len(Data_exp))
Data_set.extend(Data_exp)
Ratios.extend(Sr_exp)
print("Examples generated with Exp Decay: ", len(Data_exp))
print("Total examples: ", len(Data_set))

#2. Logistic
Data_log, Sr_log = generate_data(logistic, log_t_span[0], log_y0, log_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10, 1e-12],1)
Data_set.extend(Data_log)
Ratios.extend(Sr_log)
print("Examples generated with Logistic: ", len(Data_log))
print("Total examples: ", len(Data_set))

#3. Multiple Scales 
Data_ms, Sr_ms = generate_data(multi_scales, ms_t_span[0], ms_y0, ms_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10, 1e-12],1)
Data_set.extend(Data_ms)
Ratios.extend(Sr_ms)
print("Examples generated with Multiple Scales: ", len(Data_ms))
print("Total examples: ", len(Data_set))

#4. Oscillating
Data_o, Sr_o = generate_data(oscil, o_t_span[0], o_y0, o_t_span[1], [1e-6, 1e-8, 1e-10, 1e-12],1)
Data_set.extend(Data_o)
Ratios.extend(Sr_o)
print("Examples generated with Oscillating: ", len(Data_o))
print("Total examples: ", len(Data_set))

#5. Simple Harmonic Motion 
Data_shm, Sr_shm = generate_data(shm, shm_t_span[0], shm_y0, shm_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10], 2)
Data_set.extend(Data_shm)
Ratios.extend(Sr_shm)
print("Examples generated with SHM : ", len(Data_shm))
print("Total examples: ", len(Data_set))

#6. VanDerPol
Total_ex = 0
for f in VDP_fcns:
    Data_vdp, Sr_vdp = generate_data(f, vdp_t_span[0], vdp_y0, vdp_t_span[1], [1e-3, 1e-6, 1e-8, 1e-10], 2)
    Data_set.extend(Data_vdp)
    Ratios.extend(Sr_vdp)
    Total_ex += len(Data_vdp)

print("Examples generated with VanDerPol: ", Total_ex)
print("Total examples: ", len(Data_set))

print(f"Size of Data Matrix: {len(Data_set)}x{len(Data_set[0])}" )

# Store Dataset as a csv file
#----------------------------------------------------------------------------
with open("features2.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(Data_set)

with open("ratios2.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows([[r] for r in Ratios])

    










