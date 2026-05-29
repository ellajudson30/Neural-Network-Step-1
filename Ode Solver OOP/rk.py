import numpy as np

# class RungeKutta:
#     def init(self, A, B, C, n_stages, order):
#         self.A = A
#         self.B = B
#         self.C = C
#         #self.K = K
#         self.n_stages = n_stages
#         self.order = order

class RungeKutta:
    """Base class for explicit Runge-Kutta methods."""
    C: np.ndarray = NotImplemented
    A: np.ndarray = NotImplemented
    B: np.ndarray = NotImplemented
    order: int = NotImplemented
    n_stages: int = NotImplemented
    
    # def rk_step(self, fun, A,B,C):        # where shoudl fcn go?
    #     return y_new, f_new

class forward_euler(RungeKutta):
    A = np.array([0])
    B = np.array([1])
    C = np.array([0])
    n_stages = 1
    order = 1

class RK2(RungeKutta):
    A = np.array([[0,0], [1,0]])
    B = np.array([[0.5,0.5]])
    C = np.array([[0,1]])
    n_stages = 2
    order = 2

class RK3(RungeKutta):
    A = np.array([[0,0,0], [1,0,0],[0.25,0.25,0]])
    B = np.array([[1/6,1/6, 2/3]])
    C = np.array([[0,1, 0.5]])
    n_stages = 3
    order = 3

def rk_one_step(fcn, t, y, h, method):
    """Parameters:
    fcn - right hand side, y'= fcn(t,y)
    t - current time"
    "y - current approximation of solution at time t"\
    method - desired RK method : [forward_euler, RK2, RK3] 
    
    Returns : value of y after time step"""

    s = method.n_stages
   
    # # Initialize array to store values of each stage
    K = np.zeros(s)
    K[0] = fcn(t,y)

    # Deal with Forwards Euler Separately
    if method == forward_euler:
        y_new = y + h*K[0]
    
    else:
        for i in range(1,s):
            print('hi')
            t_temp = t + method.C[0][i]*h

            y_temp =0
            for j in range(0,i):
                
                y_temp += h*method.A[i][j]*K[j]

            K[i] = fcn(t_temp, y + y_temp)
        
        y_stages = 0
        for k in range(0,s):
            y_stages += h*method.B[0][k]*K[k]
    
        y_new = y + y_stages


    # #Compute each stage
    # for i in range(1, s):
    #     t_next = t + method.C[0][i]*h
    #     y_next = y + h*(np.dot(method.A[i], K[:s]))
    #     K[i] = fcn(t_next, y_next)

    # # Update step
    # y_new = y + h*(np.dot(method.B[0], K))

    return y_new, K