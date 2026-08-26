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
    # stage_history = []

    while solver.t < tf:
        solution.append(solver.y.copy())
        times.append(float(solver.t))
        
        t_old = solver.t
        
        solver.step()
        h = solver.t - t_old
        time_steps.append(float(h))
        # stage_history.append(solver.K.copy())

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
        np.array(solver.rejections_per_step))

# Approximate derviatives
def approx_first_deriv2(sol, times):
    """Given a sketelon of a solution, use a finite difference stencil on a 
    non-uniform grid to approximate the first derivative
    sol - skeleton of solution
    times - points where skeleton is (can use to get see timesteps taken) """
    
    y_prime = np.zeros(len(times) -2 )

    y_prime = (np.array(sol[2:]) - np.array(sol[:-2]))/(np.array(times[2:])
                                                     - np.array(times[:-2]))
    return y_prime
def approx_second_deriv2(sol, times):
    """Given a sketelon of a solution, use a finite difference stencil on a 
    non-uniform grid to approximate the second derivative 
    sol - skeleton of solution
    times - points where skeleton is (can use to get see timesteps taken) """

    y_2prime = np.zeros(len(times) -2 )

    times = np.array(times)
    sol = np.array(sol)
    
    h_left = times[1:-1] - times[:-2]
    h_right = times[2:] - times[1:-1]

    y_2prime = (2/(h_left + h_right)) * ((sol[2:] - sol[1:-1])/h_right -
                                        (sol[1:-1] - sol[:-2])/h_left)
    return y_2prime