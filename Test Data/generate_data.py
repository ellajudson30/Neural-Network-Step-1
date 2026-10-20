import numpy as np
import math
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import csv
from rk45utils import *

# ODEs used to generate data
# ------------------------------------------------------------------
# Exponential Decay
lmda = [0.1, 0.5, 1, 2, 5, 10, 50]  # increase to get more points
ed_t_span = [0,10]
ed_y0 = [0.5, 1, 1.5, 2, 3] # increase to get more points

expdec_fcns = []
for l in lmda:
    def exp_decay(t,y, l=l):
        return -l*y
    expdec_fcns.append(exp_decay)

# Logistic
R = [0.25,0.5,1,2, 5]
K = [0.5,1,2]
# log_t_span = [0,10]
# log_y0 = [0.25, 0.5, 0.75, 1.25, 2, 5] # make this a function of k to smaple to whole solution curve

log_fcns = []
for r in R:
    for k in K:
        def logistic(t,y, r=r, k=k):
            return r*y*(1-(y/k))

        log_y0 = k*np.array([0.1,0.25,0.5, 0.75, 1.25, 2, 5])
        log_t_span = [0,10/r]
        log_fcns.append((logistic, {"r": r, "k": k}, log_y0, log_t_span))

# Multiple Scales
ms_t_span = [0, 6*math.pi]
ms_y0 = [-2, -1, 0, 0.5, 1, 2]
k_vals=[1,1000]

ms_fcns = []
for k in k_vals:
    def multi_scales(t,y, k=k):
        return -k*(y-math.cos(t)) - math.sin(t)
    ms_fcns.append((multi_scales, {"k":k}))

# Oscillatory
o_t_span = [0, 6*math.pi]
o_y0 = [-2, -1, -0.5, 0.5, 1, 2]
def oscil(t,y):
    return y*math.cos(t)

# Simple Harmonic Motion
shm_t_span = [0, 6*math.pi]
shm_y0 = [[0.5, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [3.0, -1.0],]
def shm(t,Y):
    y,v = Y
    return [v,-y]

# VanDerPol
m = [0.5, 1, 2]
vdp_t_span = [0, 6*math.pi]
vdp_y0 = [[0.1, 0.0], [2.0, 0.0], [0.0, 2.0], [1.0, -1.0], [3.0, 0.0]]

VDP_fcns = []
for val in m:
    def VanDerPol(t,Y, val=val):
        y,v = Y
        return [v, val*(1-y**2)*v - y]
    VDP_fcns.append((VanDerPol,{"m": val}))

# Damped Harmonic Motion
d_t_span = [0, 6*math.pi]
d_y0 = [[0.5, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [3.0, -1.0],]
d_vals = [0.2,2,3]

d_fcns = []
for d in d_vals:
    def damped_hm(t,Y, d=d):
        y,v= Y
        return [v, -d*v - y]
    d_fcns.append((damped_hm, {"d": d}))

# ODE that blows up
bu_t_span = [0,0.95]
bu_y0 = [0.25, 0.75, 0.85,0.95, 1]
def blows_up(t,y):
    return y**2

# Chirp - frequency increases with time
chirp_t_span = [0, 10]
chirp_y0 = [-1, 0, 1]

c_vals = [0.5, 1, 1.75] # add 2,4,8
chirp_fcns=[]
for c in c_vals:
    def chirp(t, y, c=c):
        return np.cos(c*(t**2))
    chirp_fcns.append((chirp, {"c": c}))

# Gaussian Pulse - constant other than a rapid pulse at 2.5
pulse_t_span = [0, 5]
pulse_y0 = [0]

def pulse(t, y):
    return 10*np.exp(-100*(t-2.5)**2)

# Lorenz - standard parameters
sigma = 10
r_vals = [28, 317]
b = 8/3
L_t_span = [0,10]
L_y0 = [[0.0, 1.0, 20.0], [1.0, 1.0, 1.0], [0.5,0,0]]
L_fcns = []
for r in r_vals:
    def Lorenz(t,Y, r=r):
        x,y,z = Y
        return [sigma*(y-x), r*x-y-x*z, x*y-b*z]
    L_fcns.append((Lorenz, {"r": r}))


# Functions for Data Generating
#---------------------------------------------------------------------

# Function to choose data points uniformly for exp_decay
def select_points_exp(sol, num_hist=4):
    """Given a skeleton of a solution, select every 6th point for the dataset. 
    Return the indices of selected points"""

    n = len(sol)

    # Not enough history to build a feature
    min_index = num_hist + 2
    if n <= min_index + 1 :
        return []

    indices = []
    
    index = max(1, math.floor(n / 6))
    j = 1
    while j*index < n:
        idx = j*index

        if idx >= min_index:
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
# Perturbed examples: the last step is replaced by a step of size p*h (p log-uniform in
# PERTURB_RANGE) so the network also sees states where the step size is wrong.
# Label = factor the PI controller would apply after that step's error.
N_PERTURB = 2
PERTURB_RANGE = (0.5, 2.0)
perturb_rng = np.random.default_rng(0)

def build_features_solhist(indices, times, solution, full_solution, time_steps, num_hist, tol,
                           fcn, sol_component, ode_name, params, initial_condition):
    """ Builds feature vectors with build_feature (rk45utils): solution history,
    log ratio history, log h, log tol and derivative estimates up to order 5.
    For each index i, adds the example on the PI trajectory (label: PI's next ratio
    time_steps[i]/time_steps[i-1]) and N_PERTURB examples where the step into
    point i is rescaled (label: pi_factor of that step's error).
    indices - of chosen data points to be added to dataset
    num_hist - number of previous solution points (must equal NUM_HIST)
    """
    assert num_hist == NUM_HIST
    data_list = []
    next_step_ratio = []
    metadata_list = []

    # Solver object only provides the RK45 tableau and error norm for trial steps
    solver = RK45(fcn, times[0], full_solution[0], times[-1], rtol=tol, atol=tol)

    def deriv(t, y):
        return float(np.asarray(fcn(t, y)).reshape(-1)[sol_component])

    for i in indices:

        # Check if index is at the end of the time span
        if i + 1 >= len(time_steps):
            pass
        # Check if not enough solution history
        elif i < num_hist + 2:
            print("not enough solution history at this point")
        else:
            w = slice(i-num_hist, i+1)
            t_win = list(times[w])
            y_win = list(solution[w])
            d_win = [deriv(t, y) for t, y in zip(times[w], full_solution[w])]
            steps = list(time_steps[i-num_hist-2:i])

            # Example on the PI trajectory
            examples = [(1.0, build_feature(t_win, y_win, d_win, steps, tol),
                         time_steps[i]/time_steps[i-1])]

            # Perturbed examples: retake the step into point i with size p*h
            t_prev, y_prev, h = times[i-1], full_solution[i-1], time_steps[i-1]
            for p in np.exp(perturb_rng.uniform(*np.log(PERTURB_RANGE), N_PERTURB)):
                y_new, err = trial_step(solver, t_prev, y_prev, p*h)
                t_new = t_prev + p*h
                examples.append((p,
                                 build_feature(t_win[:-1] + [t_new],
                                               y_win[:-1] + [y_new[sol_component]],
                                               d_win[:-1] + [deriv(t_new, y_new)],
                                               steps[:-1] + [p*h], tol),
                                 pi_factor(err)))

            for p, feature, r_next in examples:
                data_list.append(feature)
                next_step_ratio.append(r_next)

                # Metadata corresponding to THIS example
                metadata_list.append([
                    ode_name,
                    str(initial_condition),
                    str(params),
                    i,
                    times[i],
                    tol,
                    sol_component,
                    solution[i],
                    r_next,
                    p
                ])

    return data_list, next_step_ratio, metadata_list

# Function to generate data matrix and target ratios
def generate_data(fcn, t0, y0, tf, tol, ord, ode_name=None, params = None, selection_method = 'deriv'):
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
    Meta_data = []

    for ic in y0:
        for tolerance in tol:
            if ord ==1:
                times, sol, ts,_,_,_,_,_,_ = run_RK45(fcn, t0, [ic], tf, tolerance)
            else:
                times, sol, ts,_,_,_,_,_,_ = run_RK45(fcn, t0, ic, tf, tolerance)

            # Keep full solution for feature generation
            full_sol = sol.copy()

            # Choose compnent of solution with the most variation
            # (same rule as most_varying_component, used when testing)
            max_var = int(np.argmax(np.ptp(sol, axis=0)))
            sol = sol[:,max_var]

            # Approximate derivatives to select points on solution curves 
            yp_full = compute_first_deriv(fcn, full_sol, times)
            yp = yp_full[:, max_var]

            yp2 = approx_second_deriv(sol, times)

            # Select desired points
            if selection_method == 'uniform':
                indices = select_points_exp(sol,num_hist=4)
            
            else:
                I = []

                # Find sections with largest derivs
                max_indices_yp = np.where(yp == yp.max())[0]
                max_indices_yp2 = np.where(yp2 == yp2.max())[0]

                # Find sections with smallest (most negative) derivs
                min_indices_yp = np.where(yp == yp.min())[0]
                min_indices_yp2 = np.where(yp2 == yp2.min())[0]

                # Find regions with ~0 derivative - choose point in the middle
                zero_deriv_yp = np.where(abs(yp) < 1e-4)[0]
                zero_deriv_yp2 = np.where(abs(yp2) < 1e-4)[0]

                if len(zero_deriv_yp) > 0:
                    zero_yp_rep = np.array([zero_deriv_yp[len(zero_deriv_yp)//2]], dtype=int)
                else:
                    zero_yp_rep = np.array([], dtype=int)

                if len(zero_deriv_yp2) > 0:
                    zero_yp2_rep = np.array([zero_deriv_yp2[len(zero_deriv_yp2)//2]], dtype=int)
                else:
                    zero_yp2_rep = np.array([], dtype=int)

                window = 2  # size of interval 
                min_index = 4+2
                for idx in np.concatenate((max_indices_yp, max_indices_yp2, min_indices_yp,
                                           min_indices_yp2)):
                    # min_index = 4+2

                    start = max(min_index, idx-window) # 6 = num_hist + window
                    end = min(len(sol), idx+window+1)

                    if start < end:
                        I.extend(range(start, end))
                    else:
                        I.append(min_index)
                
                # Zero regions: add representatives directly
                zero_yp_rep = zero_yp_rep[zero_yp_rep >= min_index]
                zero_yp2_rep = zero_yp2_rep[zero_yp2_rep >= min_index]

                I.extend(zero_yp_rep)
                I.extend(zero_yp2_rep)

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
            
            # # Plot selected points - check data sampling
            # selected_pts = sol[indices]
            # plt.plot(times,sol)
            # plt.plot(times[indices],selected_pts, 'o')
            # plt.show()

            # Create features/ratios and add to Dataset
            data, ratios, meta_data = build_features_solhist(indices, times, sol, full_sol, ts, 4, tolerance,
                                                              fcn, max_var, ode_name, params, ic)

            Data_set.extend(data)
            Step_ratios.extend(ratios)
            Meta_data.extend(meta_data)

    
    return Data_set, Step_ratios, Meta_data

# Check what behaviour we are sampling
#------------------------------------------------------------------------------
# f,p = VDP_fcns[1]
# Data_vdp, Sr_vdp, Md_vdp = generate_data(f, vdp_t_span[0], [[2.0, 0.0]], vdp_t_span[1], [1e-6],
#                                         2, "vanderpol", params=p)
# Data_p, Sr_p, Md_p = generate_data(pulse, pulse_t_span[0], pulse_y0, pulse_t_span[1], [1e-6],
#                                    1, "gaussian pulse")
# Data_o, Sr_o, Md_o = generate_data(oscil, o_t_span[0], [0.5], o_t_span[1], [1e-6],
#                                    1, "oscillating")
# f,p, log_y0, log_t_span = log_fcns[11]
# Data_log, Sr_log, Md_log = generate_data(f, log_t_span[0], [log_y0[0]], log_t_span[1], [1e-6],
#                                     1, 'logistic', params=p)


if __name__ == "__main__":
    # Construct Data Matrix and Target Ratios
    #------------------------------------------------------------------------

    # Generate data from above ODEs (first and second order)
    Data_set = []
    Ratios = []
    Meta_data = []

    # 1. Exp Decay
    Total_ex=0
    for l, f in zip(lmda,expdec_fcns):
        Data_exp, Sr_exp, Md_exp = generate_data(f, ed_t_span[0], ed_y0, ed_t_span[1], [1e-3,1e-4, 1e-6, 1e-8, 1e-10],
                                                1, "exp decay", params = {"lambda": l}, selection_method='uniform')
        Data_set.extend(Data_exp)
        Ratios.extend(Sr_exp)
        Meta_data.extend(Md_exp)
        Total_ex += len(Data_exp)
    print("Examples generated with Exp Decay: ", Total_ex)
    print("Total examples: ", len(Data_set))

    #2. Logistic
    Total_ex = 0
    for f, p, log_y0, log_t_span in log_fcns:
        Data_log, Sr_log, Md_log = generate_data(f, log_t_span[0], log_y0, log_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                        1, 'logistic', params=p)
        Data_set.extend(Data_log)
        Ratios.extend(Sr_log)
        Meta_data.extend(Md_log)
        Total_ex += len(Data_log)
    print("Examples generated with Logistic: ", Total_ex)
    print("Total examples: ", len(Data_set))

    #3. Multiple Scales 
    Total_ex = 0
    for f,p in ms_fcns:
        Data_ms, Sr_ms, Md_ms = generate_data(f, ms_t_span[0], ms_y0, ms_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                            1, "multi-scales", params=p)
        Data_set.extend(Data_ms)
        Ratios.extend(Sr_ms)
        Meta_data.extend(Md_ms)
        Total_ex +=len(Data_ms)
    print("Examples generated with Multiple Scales: ", Total_ex)
    print("Total examples: ", len(Data_set))

    #4. Oscillating
    Data_o, Sr_o, Md_o = generate_data(oscil, o_t_span[0], o_y0, o_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                    1, "oscillating", params={})
    Data_set.extend(Data_o)
    Ratios.extend(Sr_o)
    Meta_data.extend(Md_o)
    print("Examples generated with Oscillating: ", len(Data_o))
    print("Total examples: ", len(Data_set))

    #5. Simple Harmonic Motion 
    Data_shm, Sr_shm, Md_shm = generate_data(shm, shm_t_span[0], shm_y0, shm_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                            2, "shm", params={})
    Data_set.extend(Data_shm)
    Ratios.extend(Sr_shm)
    Meta_data.extend(Md_shm)
    print("Examples generated with SHM : ", len(Data_shm))
    print("Total examples: ", len(Data_set))

    #6. VanDerPol
    Total_ex = 0
    for f,p in VDP_fcns:
        Data_vdp, Sr_vdp, Md_vdp = generate_data(f, vdp_t_span[0], vdp_y0, vdp_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                                2, "vanderpol", params=p)
        Data_set.extend(Data_vdp)
        Ratios.extend(Sr_vdp)
        Meta_data.extend(Md_vdp)
        Total_ex += len(Data_vdp)
    print("Examples generated with VanDerPol: ", Total_ex)
    print("Total examples: ", len(Data_set))

    #7. Damped Harmonic Motion
    Total_ex=0
    for f,p in d_fcns:
        Data_d, Sr_d, Md_d = generate_data(f, d_t_span[0], d_y0, d_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                        2, "damped hm", params=p)
        Data_set.extend(Data_d)
        Ratios.extend(Sr_d)
        Meta_data.extend(Md_d)
        Total_ex += len(Data_d)
    print("Examples generated with Damped HM : ", Total_ex)
    print("Total examples: ", len(Data_set))

    #8. ODE that Blows up 
    Data_bu, Sr_bu, Md_bu = generate_data(blows_up, bu_t_span[0], bu_y0, bu_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                        1, "1d blows up", params={})
    Data_set.extend(Data_bu)
    Ratios.extend(Sr_bu)
    Meta_data.extend(Md_bu)
    print("Examples generated with y' = y**2: ", len(Data_bu))
    print("Total examples: ", len(Data_set))

    #9. Lorenz
    Total_ex = 0
    for f,p in L_fcns:
        Data_L, Sr_L, Md_L = generate_data(f, L_t_span[0], L_y0, L_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                        3, "lorenz", params=p)
        Data_set.extend(Data_L)
        Ratios.extend(Sr_L)
        Meta_data.extend(Md_L)
        Total_ex += len(Data_L)
    print("Examples generated with Lorenz: ", Total_ex)
    print("Total examples: ", len(Data_set))

    #10. Chirp
    Total_ex=0
    for f,p in chirp_fcns:
        Data_cp, Sr_cp, Md_cp = generate_data(f, chirp_t_span[0], chirp_y0, chirp_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                            1, "chirp", params=p)
        Data_set.extend(Data_cp)
        Ratios.extend(Sr_cp)
        Meta_data.extend(Md_cp)
        Total_ex += len(Data_cp)
    print("Examples generated with chirp: ", Total_ex)
    print("Total examples: ", len(Data_set))

    #11. Gaussian Pulse
    Data_p, Sr_p, Md_p = generate_data(pulse, pulse_t_span[0], pulse_y0, pulse_t_span[1], [1e-3, 1e-4, 1e-6, 1e-8, 1e-10],
                                    1, "gaussian pulse", params={})
    Data_set.extend(Data_p)
    Ratios.extend(Sr_p)
    Meta_data.extend(Md_p)
    print("Examples generated with pulse: ", len(Data_p))
    print("Total examples: ", len(Data_set))


    print(f"Size of Data Matrix: {len(Data_set)}x{len(Data_set[0])}" )

    x = np.arange(len(Ratios))
    y = np.array(Ratios)
    ones = np.ones(len(Ratios))

    plt.plot(x, y, 'o')
    plt.plot(ones)
    plt.show()

    # Store Dataset as a csv file
    #----------------------------------------------------------------------------
    with open("features_fullmeta1.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(Data_set)

    with open("ratios_fullmeta1.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows([[r] for r in Ratios])

    with open("metadata_full1.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "ODE",
            "initial_condition",
            "parameters",
            "datapoint_index",
            "time",
            "tolerance",
            "component",
            "solution",
            "true_ratio",
            "step_perturbation"
        ])

        writer.writerows(Meta_data)

    print("Data stored as csv successfully!")


    










