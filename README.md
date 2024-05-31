# Integral-Differential Equation Solver 

This repository contains the implementation of a solver for integrodifferential equations of the article: Nonlocal Lagrangian Framework for Learning Algorithm Optimization. The solver is designed to handle both first-order and second-order integrodifferential equations. 

## Objective

The main objective is to numerically solve the integrodifferential equations that continuously describe the dynamics of the ADAM optimization algorithm.

## Libraries Used

- `numpy`: For numerical operations
- `scipy`: For integration methods
- `pandas`: For handling and storing loss data
- `concurrent.futures`: For parallel execution

Install the required libraries with:
```bash
pip install numpy scipy pandas
```

## Classes and Methods

### 1. `ConvolutionIntegral`

This class handles the computation of convolution integrals using specified kernel functions.

#### Methods:

- **`__init__(alpha: float, beta: list, lambd: float)`**: Initializes the convolution integral with parameters alpha, beta, and lambda.
- **`__1rst_kernel__(t: float, tp: float, i: int) -> float`**: Computes the first-order kernel function for the convolution integral.
- **`__2nd_kernel__(t: float, tp: float, i: int) -> float`**: Computes the second-order kernel function for the convolution integral.
- **`__integral_operator__(f: callable, t: float, y_func: callable, tpmin: float, tpmax: float, second_order: bool=False) -> tuple`**: Performs the integral operation using the specified kernel function.

### 2. `AdamIDE`

This class implements the Adam-inspired solver for integrodifferential equations.

#### Methods:

- **`__init__(...)`**: Initializes the solver with various parameters such as learning rate, beta values, epsilon, etc.
- **`__df__(y: float) -> float`**: Computes the gradient of the differential equation for example 1.
- **`__loss__(y_pred: np.ndarray, y_true: np.ndarray) -> float`**: Computes the mean squared error loss.
- **`__df_MSE__(y: float) -> float`**: Computes the gradient of the mean squared error for example 2.
- **`__hat_epsilon__(t: int) -> float`**: Computes the adjusted epsilon value for numerical stability.
- **`__gamma__(t: int) -> float`**: Computes the gamma value for the optimizer.
- **`__y_dot__(t: float, y: np.ndarray) -> np.ndarray`**: Computes the first or second derivative of y for the differential equation for `solve_ivp`.
- **`optimize()`**: Optimizes the solution to the differential equation using the AdamIDE algorithm.
- **`optimize_losses()`**: Optimizes the solution and computes the loss at each time step for example 2 using the AdamIDE algorithm.

## Pseudocode

The pseucode is shown below:

### ConvolutionIntegral

```plaintext
Class ConvolutionIntegral:
    Method __init__(alpha, beta, lambd):
        Initialize alpha, beta, and lambd

    Method __1rst_kernel__(t, tp, i):
        Calculate delta_t = t - tp
        Calculate exponent
        Calculate exp_term
        Return alpha * exp_term

    Method __2nd_kernel__(t, tp, i):
        Calculate delta_t = t - tp
        Calculate exp_term
        Calculate beta_value
        If exp_term is not zero:
            If beta_value == 0.5:
                Return 2 * delta_t * exp_term
            Else if beta_value > 0.5:
                Calculate sqrt_term
                Calculate sinh_value
                If sinh_value is infinite:
                    Calculate log_exp_term
                    Calculate sinh_arg
                    Calculate log_sinh_term
                    Calculate log_values
                    Return exp(log_values)
                Else:
                    Return exp_term * sinh_value
            Else:
                Calculate sqrt_term
                Return exp_term * sin(sqrt_term * delta_t)
        Else:
            Return 0.0

    Method __integral_operator__(f, t, y_func, tpmin, tpmax, second_order):
        Set kernel to __2nd_kernel__ if second_order else __1rst_kernel__
        Define integrand_num
        Define integrand_den
        With ThreadPoolExecutor:
            Execute integrand_num
            Execute integrand_den
            Get results
        Return results
```

### AdamIDE

```plaintext
Class AdamIDE:
    Method __init__(...):
        Initialize parameters

    Method __df__(y):
        Return gradient of differential equation

    Method __loss__(y_pred, y_true):
        Return mean squared error

    Method __df_MSE__(y):
        Return gradient of mean squared error

    Method __hat_epsilon__(t):
        Return adjusted epsilon

    Method __gamma__(t):
        Return gamma value

    Method __y_dot__(t, y):
        Calculate gradient
        Define y_func for interpolation
        Perform integral operation
        Calculate F_y
        If second_order:
            Calculate dydt
            Calculate ddyddt
            Return [dydt, ddyddt]
        Else:
            Return -gamma * F_y

    Method optimize():
        Define t_eval
        Set method to 'RK45'
        Solve IVP with __y_dot__
        If verbose, print results
        Return solution

    Method optimize_losses():
        Call optimize
        Define x
        Define y_true
        For each time and y:
            Calculate y_pred
            Append loss
        Return solution and losses
```

## Contributing

Contributions to enhance the functionality or efficiency of the solvers are welcome. Please feel free to fork the repository and submit pull requests.

## Authors

- Carlos Heredia Pimienta, Ph.D - University of Barcelona
- Prof. Hidenori Tanaka - University of Harvard

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to Rhys Gould for his insightful discussions on numerical simulations.