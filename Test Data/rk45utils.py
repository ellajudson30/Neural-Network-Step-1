import math
import numpy as np
from scipy.integrate import RK45
from scipy.integrate._ivp.rk import rk_step

# Subclass of Scipy RK45 to include diagnostics
class RK45_counting(RK45):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.accepted_steps = 0
        self.rejected_steps = 0

        # Diagnostics
        self.rejections_per_step = []
        self.error_norms = []
        self.rejected_error_norms = []
        # self.rejections_this_step = 0

    def _step_impl(self):
        SAFETY = 0.9
        MIN_FACTOR = 0.2
        MAX_FACTOR = 10.0


        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False
        rejected_this_step = 0

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A,
                                   self.B, self.C, self.K)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor
                step_accepted = True
            else:
                # make changes in here to change interp? 

                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True
                self.rejected_steps += 1
                rejected_this_step += 1

                self.rejected_error_norms.append(error_norm)

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        self.accepted_steps += 1
        self.rejections_per_step.append(rejected_this_step)
        self.error_norms.append(float(error_norm))

        return True, None

# Function to run RK45
def run_RK45(fcn, t0, y0, tf, tol):
    """y' = fcn(t,y)
    time span: [t0,tf]
    y0 - initial value
    tol - tolerance (both atol and rtol for RK45 solver)"""

    solver = RK45_counting(fcn,t0, y0, tf, rtol=tol, atol=tol)
    
    # Storage arrays
    times = []
    solution = []
    time_steps = []
    errors = []

    times.append(float(solver.t))
    solution.append(solver.y.copy())

    while solver.t < tf:
        # solution.append(solver.y.copy())
        # times.append(float(solver.t))
        
        t_old = solver.t
        
        solver.step()
        h = solver.t - t_old
        time_steps.append(float(h))

        solution.append(solver.y.copy())
        times.append(float(solver.t))

        # scale = solver.atol + solver.rtol * np.maximum(np.abs(solver.y_old), 
        #                                             np.abs(solver.y))
        
        # err = solver._estimate_error_norm(solver.K, h, scale)

        # errors.append(float(err))

        errors.append(solver.error_norms[-1])

    rejected_errors = solver.rejected_error_norms

    # Compute ratios
    ts_ratios = []
    for i in range(1, len(time_steps)):
        ratio = time_steps[i]/time_steps[i-1]
        ts_ratios.append(ratio)
    return (
        np.array(times), 
        np.array(solution), 
        np.array(time_steps), 
        np.array(ts_ratios),
        np.array(errors),
        np.array(rejected_errors),
        solver.accepted_steps,
        solver.rejected_steps,
        np.array(solver.rejections_per_step)
        )

# Feature vector shared by data generation and testing
#---------------------------------------------------------------------
NUM_HIST = 4                        # history points before the current point
N_FEATURES = 2*(NUM_HIST+1) + 2 + 2*5  # sol, log ratios, log h, log tol, 5 derivs (log|.| and sign)
DERIV_FLOOR = 1e-20                 # floor for log|derivative|

def build_feature(times, sol, derivs, steps, tol):
    """Build one feature vector from windows ending at the current point.
    times, sol, derivs - NUM_HIST+1 values of t, y and y' (one solution component)
    steps - NUM_HIST+2 accepted step sizes, the last one being the step into the current point
    Returns (in order): sol history, log ratio history, log h, log tol,
    log|y^(k)| for k=1..5 at the current point, sign(y^(k)) for k=1..5.
    """
    times = np.asarray(times, dtype=float)
    steps = np.asarray(steps, dtype=float)
    h = steps[-1]

    log_ratios = np.log(steps[1:] / steps[:-1])

    # Derivatives at the current point: y' is exact, y''..y^(5) come from the
    # degree-4 polynomial through the y' history (times scaled by h for conditioning)
    s = (times - times[-1]) / h
    coeffs = np.polyfit(s, derivs, NUM_HIST)[::-1]   # coeffs[k] multiplies s^k
    d = [float(derivs[-1])]
    for k in range(1, NUM_HIST+1):
        d.append(math.factorial(k) * coeffs[k] / h**k)
    d = np.array(d)

    feature = list(sol) + list(log_ratios) + [np.log(h), np.log(tol)]
    feature += list(np.log(np.maximum(np.abs(d), DERIV_FLOOR)))
    feature += list(np.sign(d))
    return feature

def trial_step(solver, t, y, h):
    """Take one RK45 step of size h from (t, y) using the solver's tableau.
    Returns the new solution and the error norm the PI controller would see."""
    y = np.asarray(y, dtype=float)
    K = np.empty((solver.n_stages + 1, solver.n), dtype=float)
    f = np.asarray(solver.fun(t, y), dtype=float)
    y_new, _ = rk_step(solver.fun, t, y, f, h, solver.A, solver.B, solver.C, K)
    scale = solver.atol + np.maximum(np.abs(y), np.abs(y_new)) * solver.rtol
    return y_new, solver._estimate_error_norm(K, h, scale)

def pi_factor(error_norm):
    """Step-size factor the PI controller applies after a step with this error norm
    (same constants as RK45_counting._step_impl)."""
    if error_norm == 0:
        return 10.0
    return float(np.clip(0.9 * error_norm ** (-1/5), 0.2, 10.0))

# Choose the solution component used in feature vectors
def most_varying_component(fcn, t0, y0, tf, tol):
    """Run the PI solver and return the index of the solution component with the
    largest range. Training data uses this component, so testing must use it too."""
    _, sol, *_ = run_RK45(fcn, t0, y0, tf, tol)
    return int(np.argmax(np.ptp(sol, axis=0)))

# Approximate derviatives
def compute_first_deriv(fcn, sol, times):
    """Given a sketelon of a solution, use a finite difference stencil on a 
    non-uniform grid to approximate the first derivative
    sol - skeleton of solution
    times - points where skeleton is (can use to get see timesteps taken) """
    
    times = np.asarray(times)
    sol = np.asarray(sol)

    y_prime = np.array([
        np.atleast_1d(fcn(t, y))
        for t, y in zip(times, sol)
    ])
    
    return y_prime
def approx_second_deriv(sol, times):
    """Given a sketelon of a solution, use a finite difference stencil on a 
    non-uniform grid to approximate the second derivative 
    sol - skeleton of solution
    times - points where skeleton is (can use to get see timesteps taken) """

    # y_2prime = np.zeros(len(times) -2 )

    times = np.array(times)
    sol = np.array(sol).squeeze()
    
    h_left = times[1:-1] - times[:-2]
    h_right = times[2:] - times[1:-1]

    y_2prime = (2/(h_left + h_right)) * ((sol[2:] - sol[1:-1])/h_right -
                                        (sol[1:-1] - sol[:-2])/h_left)
    return y_2prime