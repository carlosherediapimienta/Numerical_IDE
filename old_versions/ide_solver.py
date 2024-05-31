import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class IDSolver:
    def __init__(self, lim, n, IC, Tol:float, Flag:int, int_partition:int, alpha:float, beta:list, epsilon:float, a:float = 0.5, max_iter:int = 20):
        self.lim = lim
        self.n = n
        self.IC = IC
        self.Tol = Tol
        self.Flag = Flag
        self.partition = int_partition
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.a = a
        self.max_iter = max_iter  

        if Flag == 0:
            self.options = {'atol': 1e-6, 'rtol': 1e-3}
            self.TolQuad = 1e-6
        elif Flag == 1:
            self.options = {'atol': 1e-8, 'rtol': 1e-8}
            self.TolQuad = 1e-8
        else:
            print('Flag = 0 or Flag = 1. No other options are supported.')
            return
    
    def __kernel__(self, i, t, s):
        delta_t = t-s
        exp_term = np.exp(-((1 - self.beta[i-1]) / self.alpha) * delta_t)
        return self.alpha * exp_term 
    
    def __df__(self, y, s):
        return 2 * (y(s) - 4.0)
    
    def __hat_epsilon__(self, t):
        return self.alpha * np.sqrt((1 - self.beta[1]**t) / (1 - self.beta[1])) * self.epsilon

    def __gamma__(self, t):
        return (1 - self.beta[0]) / (self.alpha*(1 - self.beta[0]**t)) * np.sqrt((1 - self.beta[1]**t) / (1 - self.beta[1]))

    def solve(self):
        print('\n***********************************************************************')
        print('**   IDSOLVER: A general purpose solver for nth-order ID equations   **')
        print('**   Copyright (2012) Dr. Claudio Gelmi and Dr. Héctor Jorquera      **')
        print('***********************************************************************\n')

        """
        Carlos Heredia Pimienta, Ph.D. (2024) - (******* ESCRIBIR  ********)
        """

        # Interval partition
        interval = np.linspace(self.lim[0], self.lim[1], self.partition)

        # Initial guess generator
        sol = solve_ivp(lambda t, y: self.model0(t, y, self.n), [self.lim[0], self.lim[1]], self.IC, t_eval=interval, **self.options)
        nominales = np.vstack((sol.t, sol.y[0])).T

        # Iterative solution
        error = 1e3
        iteration = 1
        print('     Error convergence\n     ')
        print(' ========================== \n')
        print(' Iteration    Error \n')
        while error > self.Tol and iteration < self.max_iter:
            sol = solve_ivp(lambda t, y: self.model(t, y, nominales, self.n, self.TolQuad), 
                            [self.lim[0], self.lim[1]], self.IC, t_eval=interval, **self.options)
            y = sol.y[0]
            error = np.sum((y - nominales[:, 1]) ** 2)

            print(f'  {iteration:4d}      {error:8.2e}')

            nominales = np.vstack((sol.t, (1 - self.a) * nominales[:, 1] + self.a * y)).T

            iteration += 1

        print('\n')
        if self.Flag == 0:
            print("Note: Problem solved using Python's default tolerances.")
        elif self.Flag == 1:
            print('Note: Problem solved using RelTol = AbsTol = 1e-8.')

        # Save and plot the final answer
        plt.plot(nominales[:, 0], nominales[:, 1])
        plt.xlabel('t')
        plt.ylabel('y')
        plt.show()

    def model0(self, t, y, n):
        dy = np.zeros(n)
        if n > 1:
            dy[:n-1] = y[1:]
        dy[n-1] = 0
        return dy

    def model(self, t, y, nominales, n, TolQuad):

        ys = lambda s: np.interp(s, nominales[:, 0], nominales[:, 1])

        dy = np.zeros(n)
        if n > 1:
            dy[:n-1] = y[1:]

        integral_numerator, _ = quad(lambda s: self.__kernel__(1, t, s) * self.__df__(ys, s), 1e-12, t, epsabs=TolQuad)
        integral_denominator, _ = quad(lambda s: self.__kernel__(2, t, s) * self.__df__(ys, s)**2, 1e-12, t, epsabs=TolQuad)

        nonlocal_part = (integral_numerator / (np.sqrt(integral_denominator) + self.__hat_epsilon__(t)))
        dy[n-1] = - self.__gamma__(t) * nonlocal_part

        return dy


t_interval = [1e-12, 25]  
n = 1              
InitCond = [0]    
Tol = 1e-8  
Flag = 0 
int_partition = int(1e3) 
alpha = 0.1
beta = [0.9, 0.25]
epsilon = 1e-8 

solver = IDSolver(t_interval, n, InitCond, Tol, Flag, int_partition, alpha, beta, epsilon)
solver.solve()
