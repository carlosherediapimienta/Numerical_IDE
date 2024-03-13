from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def sistem_first(ini_data, x) -> list:
    """
    The function `model` defines a simple first-order differential equation system.
    
    :param ec_ini: The `ec_ini` parameter represents the initial conditions of the differential equation
    system. In this specific case, it is a list containing the initial values of the dependent variable
    `y` and its derivative `v` at a specific initial point
    :param x: The `x` parameter in the `model` function represents the independent variable in the
    differential equation system. It is the variable with respect to which the differential equations
    are being solved
    :return: A list containing the derivative of y with respect to x (dydx) and the derivative of v with
    respect to x (dvdx) is being returned.
    """
    y, v = ini_data
    dydx = v
    dvdx = -y
    return [dydx, dvdx]


def sistem_second(data_ini, x, a, b, c)-> list:
    """
    This Python function calculates the derivatives of a system of second-order differential equations.
    
    :param data_ini: The `data_ini` parameter represents the initial conditions of the system. It is a
    list containing two values: the initial value of `y` (position) and the initial value of `v`
    (velocity) at the starting point of the system
    :param a: The parameter `a` represents a constant in the differential equation system
    :param b: The parameter "b" represents a constant in the differential equation that you are solving.
    It is used in the calculation of the rate of change of the second variable in the system
    :param c: The parameter `c` represents the coefficient of the variable `y` in the second equation of
    the system. It affects the rate of change of `y` in the system of differential equations you
    provided
    :return: A list containing the derivative of y with respect to x (dy/dx) and the derivative of v
    with respect to x (dv/dx) is being returned.
    """
    y, v = data_ini
    dydx = v
    dvdx = -(b/a)*v - (c/a)*y
    return [dydx, dvdx]

ini_data = [0.0, 1.0]
x = np.linspace(0, 10, 100)

sol = odeint(sistem_first, ini_data, x)

y = sol[:, 0]
plt.plot(x, y, 'b', label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ecuación diferencial')
plt.show()


# Parameter definitions:
a, b, c = 1, 0, -1 
x = np.linspace(0, 10, 100)

# Solving the differential equation system:
sol = odeint(sistem_second, ini_data, x, args=(a, b, c))
y = sol[:, 0]

plt.plot(x, y, 'b', label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ecuación diferencial')
plt.show()
