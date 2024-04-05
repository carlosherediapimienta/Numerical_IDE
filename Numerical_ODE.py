from IDE_Solver.Numerical_Solution_IDE import ide_solver_1rst_order, ide_solver_2nd_order
import numpy as np

# First order IDE - Example

def f(t, y):
    return np.sin(t) - y

def g(t, y):
    return np.exp(-t) * y

solver = ide_solver_1rst_order(f=f, g=g, y0=[0], t_span=(0, 10), a=0, b=10)
solution = solver.solve()
solver.plot_solution()

# Second order IDE - Example

def f(t, y, z):
    return -2*y + np.sin(t)

def g(t, y, z):
    return np.exp(-t) * y

y0 = 0  
z0 = 1 

solver = ide_solver_2nd_order(f, g, y0, z0, (0, 10), 0, 10)
solution = solver.solve()
solver.plot_solution()