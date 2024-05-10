import numpy as np
from scipy.integrate import solve_ivp, quad
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

class AdamIDE:
    def __init__(self, alpha=0.01, beta=[0.9, 0.999], epsilon=1e-8, t_span=(0, 10), y0=[1], example=1, y_true=None, x=None, verbose=False):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.t_span = t_span
        self.y0 = y0
        self.verbose = verbose
        self.example = example
        self.y_true = y_true
        self.x = x
        self.sol = None
        if example == 2:
            self.losses = []

    class ConvolutionIntegral:
        def __init__(self, alpha, beta):
            self.alpha = alpha
            self.beta = beta

        def G(self, t, tp, i):
            delta_t = t - tp
            return self.alpha * np.exp(-(1 - self.beta[i-1]) / self.alpha * delta_t)

        def integral_operator(self, f, t, y_func, tpmin, tpmax, i):
            integrand = lambda tp: self.G(t, tp, i) * f(y_func(tp))
            result, _ = quad(integrand, tpmin, tpmax)
            return result

    def grad_f(self, y):
        return 2 * (y - 4.0)

    def grad_f2(self, y):
        return self.grad_f(y) ** 2
    
    def calculate_loss(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)

    def update_losses(self, y_pred, y_true, time):
        self.losses.append({'Time': time, 'Loss': self.calculate_loss(y_pred, y_true)})

    def grad_mse(self, y):
        y_pred = self.x * y
        return -2 * np.mean((self.y_true - y_pred) * self.x)
    
    def grad_mse2(self, y):
        return self.grad_mse(y) ** 2

    def hat_epsilon(self, t):
        return self.alpha * np.sqrt((1 - self.beta[1]**t) / (1 - self.beta[1])) * self.epsilon
    
    def gamma(self, t):
        return (1 - self.beta[0]) / (self.alpha*(1 - self.beta[0]**t)) * np.sqrt((1 - self.beta[1]**t) / ( 1 - self.beta[1]))

    def y_dot(self, t, y):
        calculator = self.ConvolutionIntegral(self.alpha, self.beta)
        grad = self.grad_mse if self.example == 2 else self.grad_f
        grad2 = self.grad_mse2 if self.example == 2 else self.grad_f2
        y_func = lambda s: np.interp(s, self.sol.t, self.sol.y[0]) if self.sol else y[0]

        integral1 = calculator.integral_operator(grad, t, y_func, 0, t, 1)
        integral2 = calculator.integral_operator(grad2, t, y_func, 0, t, 2)

        F_y = integral1 / (np.sqrt(integral2) + self.hat_epsilon(t))
        return -self.gamma(t) * F_y

    def optimize(self):
        t_eval = np.linspace(self.t_span[0], self.t_span[1], 500)
        self.sol = solve_ivp(self.y_dot, self.t_span, self.y0, method='RK45' , t_eval=t_eval, rtol=1e-4, atol=1e-7, vectorized=False, dense_output=False)

        if self.verbose:
            print("Simulation completed. Final results:")
            print("Times:", [f"{t:.2f}" for t in self.sol.t])
            print("Theta values:", [f"{theta:.2f}" for theta in self.sol.y[0]])

        return self.sol

    def optimize_losses(self):
        self.optimize()
        self.x = np.arange(len(self.sol.t))
        self.y_true = 2 * self.x

        for t, y in zip(self.sol.t, self.sol.y[0]):
            y_pred = y * self.x
            self.update_losses(y_pred, self.y_true, t)

        return self.sol, pd.DataFrame(self.losses)

#################### IDE with second order ODE ####################

class AdamIDE2:
    def __init__(self,  alpha=0.01, beta=[0.9, 0.999], epsilon=1e-8, omega= 0, t_span=(0, 10), y0=[1, 0], example=1, y_true=None, x=None, verbose=False):
        self.a = alpha/2  # Coefficient for the second derivative
        self.b = 1  # Coefficient for the first derivative
        self.c = omega/alpha  # Coefficient for the function y
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.t_span = t_span
        self.y0 = y0
        self.verbose = verbose
        self.example = example
        self.y_true = y_true
        self.x = x
        self.sol = None
        if example == 2:
            self.losses = []

    class ConvolutionIntegral:
        def __init__(self, alpha, beta):
            self.alpha = alpha
            self.beta = beta

        def K(self, t, tp, i):
            delta_t = t - tp
            exp_term = np.exp(- delta_t / self.alpha)
            
            beta_value = self.beta[i - 1]
            
            if beta_value == 0.5:
                return 2 * delta_t * exp_term
            elif beta_value > 0.5:
                sqrt_term = np.sqrt(2 * beta_value - 1)
                return (2 * self.alpha) / sqrt_term  * exp_term * np.sinh(sqrt_term / self.alpha * delta_t)
            else: # beta_value < 0.5
                sqrt_term = np.sqrt(1 - 2 * beta_value)
                return (2 * self.alpha) / sqrt_term * exp_term * np.sin(sqrt_term / self.alpha * delta_t)

        def integral_operator(self, f, t, y_func, tpmin, tpmax, i):
            integrand = lambda tp: self.K(t, tp, i) * f(y_func(tp))
            result, _ = quad(integrand, tpmin, tpmax)
            return result

    def grad_f(self, y):
        return 2 * (y - 4.0)

    def grad_f2(self, y):
        return self.grad_f(y) ** 2
    
    def calculate_loss(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)

    def update_losses(self, y_pred, y_true, time):
        self.losses.append({'Time': time, 'Loss': self.calculate_loss(y_pred, y_true)})

    def grad_mse(self, y):
        y_pred = self.x * y
        return -2 * np.mean((self.y_true - y_pred) * self.x)
    
    def grad_mse2(self, y):
        return self.grad_mse(y) ** 2

    def hat_epsilon(self, t):
        return self.alpha * np.sqrt((1 - self.beta[1]**t) / (1 - self.beta[1])) * self.epsilon
    
    def gamma(self, t):
        return (1 - self.beta[0]) / (self.alpha*(1 - self.beta[0]**t)) * np.sqrt((1 - self.beta[1]**t) / ( 1 - self.beta[1]))

    def system_dynamics(self, t, y):
        dydt = y[1]
        calculator = self.ConvolutionIntegral(self.alpha, self.beta)
        grad = self.grad_mse if self.example == 2 else self.grad_f
        grad2 = self.grad_mse2 if self.example == 2 else self.grad_f2
        y_func = lambda s: np.interp(s, self.sol.t, self.sol.y[0]) if self.sol else y[0]

        integral1 = calculator.integral_operator(grad, t, y_func, 0, t, 1)
        integral2 = calculator.integral_operator(grad2, t, y_func, 0, t, 2)

        F_y = integral1 / (np.sqrt(integral2) + self.hat_epsilon(t))
        ddyddt = (-self.gamma(t)*F_y - self.b * y[1] - self.c * y[0]) / self.a

        return [dydt, ddyddt]

    def optimize(self):
        t_eval = np.linspace(self.t_span[0], self.t_span[1], 500)
        self.sol = solve_ivp(self.system_dynamics, self.t_span, self.y0, method='RK45',
                              t_eval=t_eval, rtol=1e-3, atol=1e-6, vectorized=False, dense_output=False)

        if self.verbose:
            print("Simulation completed. Final results:")
            print("Times:", [f"{t:.2f}" for t in self.sol.t])
            print("Theta values:", [f"{theta:.2f}" for theta in self.sol.y[0]])

        return self.sol

    def optimize_losses(self):
        self.optimize()
        self.x = np.arange(len(self.sol.t))
        self.y_true = 2 * self.x

        for t, y in zip(self.sol.t, self.sol.y[0]):
            y_pred = y * self.x
            self.update_losses(y_pred, self.y_true, t)

        return self.sol, pd.DataFrame(self.losses)
