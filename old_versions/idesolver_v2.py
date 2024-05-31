import numpy as np
from scipy.integrate import solve_ivp, fixed_quad
import warnings
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import itertools
import matplotlib.pyplot as plt
import time

warnings.filterwarnings("ignore")

class ConvolutionIntegral:
        def __init__(self, alpha, beta, lambd):
            self.alpha = alpha
            self.beta = beta
            self.lambd = lambd

        def __1rst_kernel__(self, t, tp, i):
            delta_t = t - tp
            exponente = -((1 - self.beta[i-1]) / self.alpha) * delta_t
            exp_term = np.exp(exponente)
            return self.alpha * exp_term 
                
        def __2nd_kernel__(self, t, tp, i):
            delta_t = t - tp
            exp_term = np.exp(- delta_t / self.alpha)
            beta_value = self.beta[i - 1]
            if np.all(exp_term != 0.0):
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

        def __integral_operator__(self, f, t, y_func, tpmin, tpmax, second_order=False):
            kernel = self.__2nd_kernel__ if second_order else self.__1rst_kernel__

            integrand_num = lambda tp: kernel(t, tp, 1) * (f(y_func(tp)) + self.lambd / 2 * y_func(tp))
            integrand_den = lambda tp: kernel(t, tp, 2) * (f(y_func(tp)) + self.lambd / 2 * y_func(tp))**2

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_num = executor.submit(fixed_quad, integrand_num, tpmin, tpmax, n=100)
                future_den = executor.submit(fixed_quad, integrand_den, tpmin, tpmax, n=100)

                integrand_num_result = future_num.result()[0]
                integrand_den_result = future_den.result()[0]

            return integrand_num_result, integrand_den_result

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
        self.sol = None
        self.second_order = second_order
        self.omega = omega
        self.lambd = lambd
        self.rtol = rtol
        self.atol = atol
        self.vectorized = True

        if second_order:
            self.a = alpha / 2
            self.b = 1
            self.c = omega / alpha
        if example == 2:
            if x is None:
                self.x = np.arange(1,100+1, 1)
            if y_true is None:
                self.y_true = 2*self.x
            self.losses = []

        self.calculator = ConvolutionIntegral(self.alpha, self.beta, self.lambd)

    def __df__(self, y):
        return 2 * (y - 4.0)

    def __loss__(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)
        
    def __df_MSE__(self, y):
        y_pred = y * self.x
        return -2 * np.mean((self.y_true - y_pred) * self.x)

    def __hat_epsilon__(self, t):
        return self.alpha * np.sqrt((1 - self.beta[1]**t) / (1 - self.beta[1])) * self.epsilon

    def __gamma__(self, t):
        return (1 - self.beta[0]) / (self.alpha*(1 - self.beta[0]**t)) * np.sqrt((1 - self.beta[1]**t) / (1 - self.beta[1]))

    def __y_dot__(self, t, y):
        grad = self.__df_MSE__ if self.example == 2 else self.__df__
        y_func = lambda s: np.interp(s, self.sol.t, self.sol.y[0]) if self.sol else y[0]

        integral_num, integral_den = self.calculator.__integral_operator__(grad, t, y_func, self.t_0, t, second_order=self.second_order)
        F_y = integral_num / (np.sqrt(integral_den) + self.__hat_epsilon__(t))

        if self.second_order:
            dydt = y[1]
            ddyddt = (-self.__gamma__(t)*F_y - self.b * y[1] - self.c * y[0]) / self.a 
            return [dydt, ddyddt]
        else:
            return -self.__gamma__(t) * F_y

    def optimize(self):
        t_eval = np.linspace(self.t_span[0], self.t_span[1], 800)
        method = 'RK45'
        self.sol = solve_ivp(self.__y_dot__, self.t_span, self.y0, method=method, t_eval=t_eval,\
                             rtol=self.rtol, atol=self.atol, vectorized=self.vectorized, dense_output=True)

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
            self.losses.append({'Time': t, 'Loss': self.__loss__(y_pred, self.y_true)})

        return self.sol, pd.DataFrame(self.losses)
    

param_grid = {
    'lr': [0.1],
    'beta1': [0.9],
    'beta2': [0.25],
    }

n_learning_rates = len(param_grid['lr'])
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

np.random.seed(33)
theta_initial = np.random.uniform(2,3)
theta_initial_dot = np.random.uniform(2,3)

t_max = 1
t_0 = 1e-12

example = 1
verbose = False

fig, axs = plt.subplots(ncols=n_learning_rates, figsize=(15, 8), sharex=False, sharey=True)  
fig.suptitle('Convergence Trajectories for Different second-order nonlocal continuous Adam Configurations')

## Initial value for theta and theta dot
y0 = [theta_initial,  theta_initial_dot]

## IVPSolver configuration
if example == 1:
    rtol = 5e-2
    atol = 1e-4 
elif example == 2:
    rtol = 5e-2
    atol = 1e-4 

for i, lr in enumerate(param_grid['lr']):

    ax = axs[i] if n_learning_rates > 1 else axs

    filtered_params = [p for p in all_params if p['lr'] == lr]

    for params in filtered_params:

        start_time = time.time()

        print(f'\nNonlocal continuous Adam Configuration: {params}')

        optimizer = AdamIDE(second_order=True, t_max=t_max, t_0=t_0, alpha=lr, beta=[params['beta1'], params['beta2']], 
                             y0=y0, example=example,  rtol=rtol, atol=atol, verbose=verbose)
        
        if example == 1:
            sol = optimizer.optimize()
        elif example == 2:
            sol, losses_IDE = optimizer.optimize_losses()

        elapsed_time = (time.time() - start_time)/60

        print(f"Final parameter value ADAMIDE: theta = {sol.y[0][-1]:.4f}, with elapsed time: {elapsed_time:.2f} minutes.")

        label = f"beta1={params['beta1']}, beta2={params['beta2']}"
        if example == 1:
            ax.plot(sol.t, sol.y[0], label=label)
            ax.set_ylabel('Theta value')
        elif example == 2:
            ax.plot(losses_IDE['Time'], losses_IDE['Loss'], label=label)
            ax.set_ylabel('Loss')
        ax.set_xlim([t_0, t_max])
        ax.set_title(f'Learning Rate = {lr}')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()