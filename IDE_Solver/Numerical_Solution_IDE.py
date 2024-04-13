import numpy as np
from scipy.integrate import quad, solve_ivp
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

class AdamOptimizerIDE:
    def __init__(self, alpha=0.01, beta=[0.9, 0.999], epsilon=1e-8, t_span=(0, 10), theta0=[1]):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.t_span = t_span
        self.theta = np.array(theta0,dtype=np.float64)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.sol = None

    class ConvolutionIntegral:
        def __init__(self, f, f2, alpha, beta):
            self.f = f 
            self.f2 = f2
            self.alpha = alpha
            self.beta = beta

        def G(self, tp, t, i):
            return np.exp(((1 - self.beta[i-1]) / self.alpha) * (tp - t))

        def integral_operator(self, t, theta_func, tpmin, tpmax, i):
            integrand = lambda tp: self.G(tp, t, i) * (self.f if i == 1 else self.f2)(theta_func(tp))
            result, error = quad(integrand, tpmin, tpmax)
            return result

    def grad_f(self, theta):
        return 2 * (theta - 4.0)

    def grad_f2(self, theta):
        return self.grad_f(theta) ** 2

    def hat_epsilon(self, t):
        return np.sqrt(self.alpha * (1 - self.beta[1]**t) / (1 - self.beta[1])) * self.epsilon

    def gamma(self, t):
        return ((1 - self.beta[0]) / (1 - self.beta[0]**t)) * np.sqrt((1 - self.beta[1]**t) / (self.alpha * (1 - self.beta[1])))

    def theta_dot(self, t, theta):
        calculator = self.ConvolutionIntegral(self.grad_f, self.grad_f2, self.alpha, self.beta)
        theta_func = lambda s: np.interp(s, self.sol.t, self.sol.y[0]) if self.sol is not None else self.theta
        futures = [self.executor.submit(calculator.integral_operator, t, theta_func, 0, t, i) for i in [1, 2]]
        results = [future.result() for future in futures]
        F_theta = results[0] / (np.sqrt(results[1]) + self.hat_epsilon(t))
        gamma_val = self.gamma(t)
        theta_dot_val = -gamma_val * F_theta
        return theta_dot_val

    def optimize(self):
        self.sol = solve_ivp(self.theta_dot, self.t_span, self.theta, method='RK45', dense_output=True, vectorized=False)
        print("Simulation completed. Final results:")
        print(f"Times: {[f'{t:.2f}' for t in self.sol.t]}")
        print(f"Theta values: {[f'{theta:.2f}' for theta in self.sol.y[0]]}")

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