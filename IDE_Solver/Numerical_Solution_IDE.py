import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from concurrent.futures import ThreadPoolExecutor
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

class AdamOptimizerIDE:
    def __init__(self, alpha=0.01, beta =[0.9,0.999], epsilon=1e-8, t_span=(0, 10), y0=[1], example=1, y_true = None, x= None, verbose=False):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.t_span = t_span
        self.y0 = y0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.sol = None
        self.verbose = verbose
        self.example = example
        self.y_true = y_true
        self.x = x
        
        if example == 2:
            self.losses = pd.DataFrame()

    class ConvolutionIntegral:
        def __init__(self, f, f2, alpha, beta):
            self.f = f 
            self.f2 = f2
            self.alpha = alpha
            self.beta = beta

        def G(self, tp, t, i):
            return np.exp(((1 - self.beta[i-1]) / self.alpha) * (tp - t))

        def integral_operator(self, t, y_func, tpmin, tpmax, i):
            integrand = lambda tp: self.G(tp, t, i) * (self.f if i == 1 else self.f2)(y_func(tp))
            result, _ = quad(integrand, tpmin, tpmax)
            return result

    def grad_f(self, y):
        result = 2 * (y - 4.0)
        return result

    def grad_f2(self, y):
        result = self.grad_f(y) ** 2
        return result
    
###### LOSS FUNCTION ######    
    
    def calculate_loss(self, y_pred, y_true):
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def update_losses(self, y_pred, y_true, time):
        loss = self.calculate_loss(y_pred, y_true)
        new_entry = pd.DataFrame({'Time': [time], 'Loss': [loss]})
        self.losses = pd.concat([self.losses, new_entry], ignore_index=True)

####### LOSS FUNCTION #######

    def grad_mse(self, y):
        y_pred = self.x * y
        return -2 * np.mean((self.y_true - y_pred) * self.x)
    
    def grad_mse2(self, y):
        result = self.grad_mse(y) ** 2
        return result

    def hat_epsilon(self, t):
        aux_1 = (self.alpha * (1 - self.beta[1]**t)) / (1 - self.beta[1])
        return np.sqrt(aux_1) * self.epsilon
    
    def gamma(self, t):
        aux_1 = (1 - self.beta[0]) / (1 - self.beta[0]**t)
        aux_2 = np.sqrt((1 - self.beta[1]**t) / (self.alpha * (1 - self.beta[1])),dtype=np.float64)
        return aux_1 * aux_2
    
    def y_dot(self, t, y):

        if self.example == 1:
            calculator = self.ConvolutionIntegral(self.grad_f, self.grad_f2, self.alpha, self.beta)
        elif self.example == 2:
            calculator = self.ConvolutionIntegral(self.grad_mse, self.grad_mse2, self.alpha, self.beta)

        y_func = lambda s: np.interp(s, self.sol.t, self.sol.y[0]) if self.sol else y[0]
        futures = [self.executor.submit(calculator.integral_operator, t, y_func, 0, t, i) for i in [1, 2]]
        results = [future.result() for future in futures]

        F_y = results[0] / (np.sqrt(results[1]) + self.hat_epsilon(t))
        gamma_val = self.gamma(t)
        y_dot_val = -gamma_val * F_y

        return y_dot_val

    def optimize(self):
        self.sol = solve_ivp(self.y_dot, self.t_span, self.y0, method='RK45', dense_output=True, vectorized=False)

        if self.verbose:
            print("Simulation completed. Final results:")
            print(f"Times: {[f'{t:.2f}' for t in self.sol.t]}")
            print(f"Theta values: {[f'{theta:.2f}' for theta in self.sol.y[0]]}")

        return self.sol
    
    def optimize_losses(self):
        self.sol = solve_ivp(self.y_dot, self.t_span, self.y0, method='RK45', dense_output=True, vectorized=False)

        if self.verbose:
            print("Simulation completed. Final results:")
            print(f"Times: {[f'{t:.2f}' for t in self.sol.t]}")
            print(f"Theta values: {[f'{theta:.2f}' for theta in self.sol.y[0]]}")
               

        self.y_true = 2*np.arange(len(self.sol.t))
        self.x = np.arange(len(self.sol.t))
       

        for t, y in zip(self.sol.t, self.sol.y[0]):
            y_p = y * self.x
            self.update_losses(y_p, self.y_true, t)

        return self.sol, self.losses





# The `ide_solver_1rst_order` class is designed to solve and plot the numerical solution of a
# 2nd-order integral-differential equation.
class ide_solver_2nd_order:
    def __init__(self, f, g, y0, z0, t_span, a, b):

        self.f = f
        self.g = g
        self.y0 = [y0, z0] 
        self.t_span = t_span
        self.a = a
        self.b = b

    def integral_part(self, t, y, z):
        result, _ = quad(self.g, self.a, self.b, args=(y, z))
        return result

    def dydt(self, t, yz):
        y, z = yz  
        integral = self.integral_part(t, y, z)
        return [z, self.f(t, y, z) + integral] 

    def solve(self):
        self.solution = solve_ivp(self.dydt, self.t_span, self.y0, method='RK45')
        return self.solution