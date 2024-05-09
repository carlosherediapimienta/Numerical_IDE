import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def grad_mse(theta, x, y_true):
    y_pred = x * theta
    return -2 * np.mean((y_true - y_pred) * x)

def grad_f(theta)->float:
    return 2 * (theta - 4)

def test_adam_convergence(epochs:int=10000, cutoff:float=0.001, theoretical_result:float=4.0, theta:float=0.1) -> bool:
    optimizer = Adam()  
    for _ in range(epochs):  
        grad = grad_f(theta)
        theta = optimizer.update(grad, theta)
        if np.abs(theta - theoretical_result) < cutoff:
            print(f"\nConverged to {theta} after {_+1} iterations.")
            return True
    print(f"\nDid not converge, final theta is {theta}.")
    return False


class Adam:
    def __init__(self, lr:float=0.01, beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, grad, theta):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return theta
