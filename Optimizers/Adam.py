import numpy as np

def grad_f(theta):
    return 2*(theta-4)


def adam_optimization(grad, theta, lr:float=0.001, beta1:float=0.9, beta2:float=0.999, 
                      epsilon:float=1e-8, t:float=0.0, m:float=0.0, v:float=0.0):
    """
    Updates the parameter theta using the Adam optimization algorithm based on the current gradient.

    Parameters:
    - grad: Gradient of the loss function with respect to theta at time t.
    - theta: Current value of the parameter being optimized.
    - lr: Learning rate.
    - beta1, beta2: Coefficients for estimating first and second order moments.
    - epsilon: Smoothing term to prevent division by zero.
    - t: Current iteration number.
    - m, v: Estimated first and second order moments of the gradients up to now.

    Returns:
    - theta: Updated parameter.
    - m, v: Updated moments.
    - t: Updated time step counter.
    """

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)

    return theta, m, v
