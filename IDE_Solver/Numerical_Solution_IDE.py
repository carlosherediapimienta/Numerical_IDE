from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt

# The `ide_solver_1rst_order` class is designed to solve and plot the numerical solution of a
# first-order integral-differential equation.
class ide_solver_1rst_order:
    def __init__(self, f, g, y0, t_span, a, b) -> None:

        self.f = f
        self.g = g
        self.y0 = y0
        self.t_span = t_span
        self.a = a
        self.b = b
        self.solution = None
    
    def integral_part(self, t, y):
        result, _ = quad(self.g, self.a, self.b, args=(y,))
        return result

    def dydt(self, t, y):
        return self.f(t, y) + self.integral_part(t, y)
    
    def solve(self):
        self.solution = solve_ivp(self.dydt, self.t_span, self.y0, method='RK45')
        return self.solution
    
    def plot_solution(self):
        if self.solution is None:
            print("The solution has not been calculated yet. Call .solve() first.")
            return
        plt.plot(self.solution.t, self.solution.y[0], label='y(t)')
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.title('1st-order Numerical IDE Solution.')
        plt.legend()
        plt.show()

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

    def plot_solution(self):
        if self.solution is None:
            print("The solution has not been calculated yet. Call .solve() first.")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(self.solution.t, self.solution.y[0], label='y(t)')
        plt.plot(self.solution.t, self.solution.y[1], label="y'(t)", linestyle='--')
        plt.xlabel('t')
        plt.ylabel('y(t), y\'(t)')
        plt.title('2nd-order Numerical IDE Solution.')
        plt.legend()
        plt.show()