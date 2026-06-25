import numpy as np
import math
import random 
from scipy.integrate import RK45
import torch
from torch.utils.data import Dataset

# Exponential Decay
lmda = 0.5
ed_t_span = [0,5]
ed_y0 = [1]
def exp_decay(t,y):
    return -lmda*y

# Logistic
log_t_span = [0,5]
log_y0 = [0.125, 0.25, 0.5, 0.75, 1, 2] 
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

# Function to run RK45
def run_RK45(fcn, t0, y0, tf, tol):

    solver = RK45(fcn,t0, y0, tf, rtol=tol, atol=tol)
    
    # Storage arrays
    times = []
    solution = []
    time_steps = []
    errors = []
    stage_history = []

    while solver.t < tf:
        solution.append(solver.y[0]) # only works if one IC passed
        times.append(float(solver.t))
        
        t_old = solver.t
        
        solver.step()
        h = solver.t - t_old
        time_steps.append(float(h))
        stage_history.append(solver.K.copy())

        scale = solver.atol + solver.rtol * np.maximum(np.abs(solver.y_old), 
                                                    np.abs(solver.y))
        
        err = solver._estimate_error_norm(solver.K, h, scale)

        errors.append(float(err))

    return times, solution, time_steps, errors, stage_history

# Approximate derviatives
def approx_first_deriv2(sol, times):
    y_prime = np.zeros(len(times) -2 )

    y_prime = (np.array(sol[2:]) - np.array(sol[:-2]))/(np.array(times[2:])
                                                     - np.array(times[:-2]))
    return y_prime
def approx_second_deriv2(sol, times):
    y_2prime = np.zeros(len(times) -2 )

    times = np.array(times)
    sol = np.array(sol)
    
    h_left = times[1:-1] - times[:-2]
    h_right = times[2:] - times[1:-1]

    y_2prime = (2/(h_left + h_right)) * ((sol[2:] - sol[1:-1])/h_right -
                                        (sol[1:-1] - sol[:-2])/h_left)
    return y_2prime

# Function to choose data points uniformly for exp_decay
def select_points_exp(sol):
    indices = []
    n = len(sol)
    index = math.floor(n/6)
    j = 1
    while j*index < n:
        indices.append(j*index)
        j += 1

    return indices

def build_features(indices, solution, time_steps, stage_history):

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

# Function to generate data matrix and target ratios
def generate_data(fcn, t0, y0, tf, tol):
    Data_set = []
    Step_ratios = []
    #I_total = 0 

    for i in y0:
        for j in tol:

            times, sol, ts, err, sh = run_RK45(fcn, t0, [i], tf, [j])
            yp = approx_first_deriv2(sol, times)
            yp2 = approx_second_deriv2(sol, times)
            n = len(yp)

            # Select desired points
            if fcn == exp_decay:
                indices = select_points_exp(sol)
            
            else:
                I = []

                # Find sections with largest derivs
                max_indices_yp = np.where(yp == yp.max())[0]
                max_indices_yp2 = np.where(yp2 == yp2.max())[0]

                # Find sections with ~0 derivative
                zero_deriv_yp = np.where(abs(yp) < 1e-4)[0]
                zero_deriv_yp2 = np.where(abs(yp2) < 1e-4)[0]

                # Choose some sections at random
                rand_points = random.sample(range(1, len(sol)-1), k=3)
                # if points are same as others no new data is added for boring/normal section

                window = 2  # size of interval 
                for idx in np.concatenate((max_indices_yp, max_indices_yp2, zero_deriv_yp,
                                           zero_deriv_yp2, rand_points)):
                    start = max(1, idx-window) 
                    #exclude 1 to avoid large ratio from chosen first step?????
                    
                    end = min(len(sol), idx+window+1)

                    I.extend(range(start, end))

                indices = np.unique(I)
                # I_total += len(indices)

            # Create features/ratios and add to Dataset
            data, ratios = build_features(indices, sol, ts, sh)

            Data_set.extend(data)
            Step_ratios.extend(ratios)

    
    return Data_set, Step_ratios

# Create data set from 4 first order ODEs
Data_set = []
Ratios = []

# 1. Exp Decay
Data_exp, Sr_exp = generate_data(exp_decay, ed_t_span[0], ed_y0, ed_t_span[1], [1e-6, 1e-8, 1e-10])
print("Examples generated with Exp Decay: ", len(Data_exp))
Data_set.extend(Data_exp)
Ratios.extend(Sr_exp)
print("Examples generated with Exp Decay: ", len(Data_exp))
print("Total examples: ", len(Data_set))

#2. Logistic
Data_log, Sr_log = generate_data(logistic, log_t_span[0], log_y0, log_t_span[1], [1e-6, 1e-8, 1e-10])
Data_set.extend(Data_log)
Ratios.extend(Sr_log)
print("Examples generated with Logistic: ", len(Data_log))
print("Total examples: ", len(Data_set))

#3. Multiple Scales 
Data_ms, Sr_ms = generate_data(multi_scales, ms_t_span[0], ms_y0, ms_t_span[1], [1e-6, 1e-8, 1e-10])
Data_set.extend(Data_ms)
Ratios.extend(Sr_ms)
print("Examples generated with Multiple Scales: ", len(Data_ms))
print("Total examples: ", len(Data_set))

#4. Oscillating
Data_o, Sr_o = generate_data(oscil, o_t_span[0], o_y0, o_t_span[1], [1e-6, 1e-8, 1e-10])
Data_set.extend(Data_o)
Ratios.extend(Sr_o)
print("Examples generated with Oscillating: ", len(Data_o))
print("Total examples: ", len(Data_set))

print(f"Size of Data Matrix: {len(Data_set)}x{len(Data_set[0])}" )

# Create Pytorch Dataset 
class myDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x,y

# Convert Data to torch tensor
D = torch.tensor(Data_set)
R = torch.tensor(Ratios, dtype=torch.long)

# Best way to split into training and testing sets? Taking last n rows will just be oscillating data



