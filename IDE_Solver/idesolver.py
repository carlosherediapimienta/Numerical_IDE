import numpy as np
from scipy.integrate import solve_ivp, fixed_quad
import warnings
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Disable warnings for cleaner output
warnings.filterwarnings("ignore")

class ConvolutionIntegral:
    def __init__(self, alpha: float, beta: list, lambd: float):
        """
        Initialize the convolution integral with parameters alpha, beta, and lambda.
        
        Parameters:
        alpha (float): Parameter for kernel functions.
        beta (list): List of beta values for the kernel functions.
        lambd (float): Regularization parameter.
        """
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd

    def __1rst_kernel__(self, t: float, tp: float, i: int) -> float:
        """
        Calculate the first kernel function for the convolution integral.
        
        Parameters:
        t (float): Current time.
        tp (float): Previous time.
        i (int): Index for beta value selection.

        Returns:
        float: Value of the first kernel function.
        """
        delta_t = t - tp
        exponent = -((1 - self.beta[i-1]) / self.alpha) * delta_t
        exp_term = np.exp(exponent)
        return self.alpha * exp_term

    def __2nd_kernel__(self, t: float, tp: float, i: int) -> float:
        """
        Calculate the second kernel function for the convolution integral.
        
        Parameters:
        t (float): Current time.
        tp (float): Previous time.
        i (int): Index for beta value selection.

        Returns:
        float: Value of the second kernel function.
        """
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

    def __integral_operator__(self, f: callable, t: float, y_func: callable, tpmin: float, tpmax: float, second_order: bool=False) -> tuple:
        """
        Perform the integral operation using the specified kernel function.
        
        Parameters:
        f (callable): Function representing the differential equation.
        t (float): Current time.
        y_func (callable): Function interpolating the solution y at time tp.
        tpmin (float): Minimum time for integration.
        tpmax (float): Maximum time for integration.
        second_order (bool): Flag to use the second-order kernel function.

        Returns:
        tuple: Results of the numerator and denominator of the integral operation.
        """
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
    def __init__(self, alpha: float=0.01, beta: list=None, epsilon: float=1e-8, t_max: float=15, t_0: float=1e-12, y0: list=[1], example: int=1,
                 y_true: np.ndarray=None, x: np.ndarray=None, verbose: bool=False, second_order: bool=False, omega: float=0, lambd: float=0, rtol: float=1e-3, atol: float=1e-6):
        """
        Initialize the AdamIDE solver with various parameters.
        
        Parameters:
        alpha (float): Learning rate.
        beta (list): Beta parameters for the optimizer.
        epsilon (float): Small constant for numerical stability.
        t_max (float): Maximum time for integration.
        t_0 (float): Initial time for integration.
        y0 (list): Initial condition for the solution.
        example (int): Example number for the differential equation.
        y_true (np.ndarray): True values for the solution (for example 2).
        x (np.ndarray): Input values for example 2.
        verbose (bool): Flag for verbose output.
        second_order (bool): Flag for second-order differential equations.
        omega (float): Frequency parameter for second-order equations.
        lambd (float): Regularization parameter.
        rtol (float): Relative tolerance for the solver.
        atol (float): Absolute tolerance for the solver.
        """
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
                self.x = np.arange(1, 101, 1)
            if y_true is None:
                self.y_true = 2 * self.x

            self.losses = []

        self.calculator = ConvolutionIntegral(self.alpha, self.beta, self.lambd)

    def __df__(self, y: float) -> float:
        """
        Compute the gradient of the differential equation for example 1.
        
        Parameters:
        y (float): Current value of y.

        Returns:
        float: Gradient of the differential equation.
        """
        return 2 * (y - 4.0)

    def __loss__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the mean squared error loss.
        
        Parameters:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.

        Returns:
        float: Mean squared error loss.
        """
        return np.mean((y_true - y_pred) ** 2)
        
    def __df_MSE__(self, y: float) -> float:
        """
        Compute the gradient of the mean squared error for example 2.
        
        Parameters:
        y (float): Current value of y.

        Returns:
        float: Gradient of the mean squared error.
        """
        y_pred = y * self.x
        return -2 * np.mean((self.y_true - y_pred) * self.x)

    def __hat_epsilon__(self, t: int) -> float:
        """
        Compute the adjusted epsilon value for numerical stability.
        
        Parameters:
        t (int): Current iteration number.

        Returns:
        float: Adjusted epsilon value.
        """
        return self.alpha * np.sqrt((1 - self.beta[1] ** t) / (1 - self.beta[1])) * self.epsilon

    def __gamma__(self, t: int) -> float:
        """
        Compute the gamma value for the optimizer.
        
        Parameters:
        t (int): Current iteration number.

        Returns:
        float: Gamma value.
        """
        return (1 - self.beta[0]) / (self.alpha * (1 - self.beta[0] ** t)) * np.sqrt((1 - self.beta[1] ** t) / (1 - self.beta[1]))

    def __y_dot__(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of y for the differential equation.
        
        Parameters:
        t (float): Current time.
        y (np.ndarray): Current value of y.

        Returns:
        np.ndarray: Derivative of y.
        """
        grad = self.__df_MSE__ if self.example == 2 else self.__df__
        y_func = lambda s: np.interp(s, self.sol.t, self.sol.y[0]) if self.sol else y[0]

        integral_num, integral_den = self.calculator.__integral_operator__(grad, t, y_func, self.t_0, t, second_order=self.second_order)
        F_y = integral_num / (np.sqrt(integral_den) + self.__hat_epsilon__(t))

        if self.second_order:
            dydt = y[1]
            ddyddt = (-self.__gamma__(t) * F_y - self.b * y[1] - self.c * y[0]) / self.a 
            return [dydt, ddyddt]
        else:
            return -self.__gamma__(t) * F_y

    def optimize(self):
        """
        Optimize the solution to the differential equation using the AdamIDE algorithm.
        
        Returns:
        OdeResult: Solution object containing the results of the integration.
        """
        t_eval = np.linspace(self.t_span[0], self.t_span[1], 800)
        method = 'RK45'
        self.sol = solve_ivp(self.__y_dot__, self.t_span, self.y0, method=method, t_eval=t_eval,
                             rtol=self.rtol, atol=self.atol, vectorized=self.vectorized, dense_output=True)

        if self.verbose:
            print("Simulation completed. Final results:")
            print("Times:", [f"{t:.2f}" for t in self.sol.t])
            print("Theta values:", [f"{theta:.2f}" for theta in self.sol.y[0]])

        return self.sol

    def optimize_losses(self):
        """
        Optimize the solution and compute the loss at each time step for example 2.
        
        Returns:
        tuple: Solution object and DataFrame containing loss values.
        """
        self.optimize()
        self.x = np.linspace(0, 1, len(self.sol.t))
        self.y_true = 2 * self.x

        for t, y in zip(self.sol.t, self.sol.y[0]):
            y_pred = y * self.x
            self.losses.append({'Time': t, 'Loss': self.__loss__(y_pred, self.y_true)})

        return self.sol, pd.DataFrame(self.losses)
