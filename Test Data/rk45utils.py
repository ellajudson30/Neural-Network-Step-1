import numpy as np
from scipy.integrate import RK45

# Function to run RK45
def run_RK45(fcn, t0, y0, tf, tol):
    """y' = fcn(t,y)
    time span: [t0,tf]
    y0 - initial value
    tol - tolerance (both atol and rtol for RK45 solver)"""

    solver = RK45(fcn,t0, y0, tf, rtol=tol, atol=tol)
    
    # Storage arrays
    times = []
    solution = []
    time_steps = []
    errors = []
    stage_history = []

    while solver.t < tf:
        solution.append(solver.y.copy())
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

    return (
        np.array(times), 
        np.array(solution), 
        np.array(time_steps), 
        np.array(errors), 
        stage_history)

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