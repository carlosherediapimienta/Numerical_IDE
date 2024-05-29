import numpy as np
from scipy.integrate import solve_ivp, quad
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

class AdamIDE:
    def __init__(self, alpha=0.01, beta=None, epsilon=1e-8, t_max= 15, t_0 = 1e-12, y0=[1], example=1,
                  y_true=None, x=None, verbose=False, second_order=False, omega=0, lambd = 0, rtol=1e-3, atol=1e-6):
        
        if beta is None:
            beta = [0.9, 0.999]
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.t_max = t_max
        self.t_0 = t_0
        self.t_span = (t_0, t_max)
        self.y0 = y0
        self.verbose = verbose
        self.example = example
        self.y_true = y_true
        self.x = x
        self.sol = None
        self.second_order = second_order
        self.omega = omega
        self.lambd = lambd
        self.rtol = rtol
        self.atol = atol
        if second_order:
            self.a = alpha / 2
            self.b = 1
            self.c = omega / alpha
        if example == 2:
            self.losses = []

    class ConvolutionIntegral:
        def __init__(self, alpha, beta, lambd):
            self.alpha = alpha
            self.beta = beta
            self.lambd = lambd

        def G(self, t, tp, i):
            delta_t = t - tp
            exp_term = np.exp(-((1 - self.beta[i-1]) / self.alpha) * delta_t)
            return self.alpha * exp_term if exp_term != 0.0 else 0.0
                

        def K(self, t, tp, i):
            delta_t = t - tp
            exp_term = np.exp(- delta_t / self.alpha)
            beta_value = self.beta[i - 1]
            if exp_term != 0.0:
                if beta_value == 0.5:
                    return 2 * delta_t * exp_term
                elif beta_value > 0.5:
                    sqrt_term = np.sqrt(2 * beta_value - 1)
                    sinh_value = np.sinh(sqrt_term / self.alpha * delta_t)
                    if np.any(np.isinf(sinh_value)):
                        log_exp_term = np.log(exp_term)

                        sinh_arg = sqrt_term / self.alpha * delta_t 
                        log_sinh_term = sinh_arg - np.log(2)

                        log_values = log_exp_term + log_sinh_term
                        return (2 * self.alpha) / sqrt_term * np.exp(log_values)                
                    else:
                        return (2 * self.alpha) / sqrt_term * exp_term * np.sinh(sqrt_term / self.alpha * delta_t)
                else:
                    sqrt_term = np.sqrt(1 - 2 * beta_value)
                    return (2 * self.alpha) / sqrt_term * exp_term * np.sin(sqrt_term / self.alpha * delta_t)
            else:
                return 0.0

        def integral_operator(self, f, t, y_func, tpmin, tpmax, i, second_order=False):
            kernel = self.K if second_order else self.G
            integrand = lambda tp: kernel(t, tp, i) * (f(y_func(tp)) + self.lambd / 2 * y_func(tp))
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
        y_pred = y * self.x
        return -2 * np.mean((self.y_true - y_pred) * self.x)

    def grad_mse2(self, y):
        return self.grad_mse(y) ** 2

    def hat_epsilon(self, t):
        return self.alpha * np.sqrt((1 - self.beta[1]**t) / (1 - self.beta[1])) * self.epsilon

    def gamma(self, t):
        return (1 - self.beta[0]) / (self.alpha*(1 - self.beta[0]**t)) * np.sqrt((1 - self.beta[1]**t) / (1 - self.beta[1]))

    def y_dot(self, t, y):
        grad = self.grad_mse if self.example == 2 else self.grad_f
        grad2 = self.grad_mse2 if self.example == 2 else self.grad_f2
        y_func = lambda s: np.interp(s, self.sol.t, self.sol.y[0]) if self.sol else y[0]

        integral1 = self.calculator.integral_operator(grad, t, y_func, self.t_0, t, 1, second_order=self.second_order)
        integral2 = self.calculator.integral_operator(grad2, t, y_func, self.t_0, t, 2, second_order=self.second_order)

        F_y = integral1 / (np.sqrt(integral2) + self.hat_epsilon(t))
        print(t, integral1, integral2, F_y, y[0])
        if self.second_order:
            dydt = y[1]
            ddyddt = (-self.gamma(t)*F_y - self.b * y[1] - self.c * y[0]) / self.a 
            return [dydt, ddyddt]
        else:
            return -self.gamma(t) * F_y

    def optimize(self):
        t_eval = np.linspace(self.t_span[0], self.t_span[1], 500)
        method = 'RK45'
        self.calculator = self.ConvolutionIntegral(self.alpha, self.beta, self.lambd)
        self.sol = solve_ivp(self.y_dot, self.t_span, self.y0, method=method, t_eval=t_eval, rtol=self.rtol, atol=self.atol, vectorized=False, dense_output=False)

        if self.verbose:
            print("Simulation completed. Final results:")
            print("Times:", [f"{t:.2f}" for t in self.sol.t])
            print("Theta values:", [f"{theta:.2f}" for theta in self.sol.y[0]])

        return self.sol

    def optimize_losses(self):
        self.optimize()
        self.x = np.linspace(0, 1, len(self.sol.t))
        self.y_true = 2 * self.x

        for t, y in zip(self.sol.t, self.sol.y[0]):
            y_pred = y * self.x
            self.update_losses(y_pred, self.y_true, t)

        return self.sol, pd.DataFrame(self.losses)