from Optimizers.Adam import grad_f, AdamOptimizer, test_adam_convergence
from IDE_Solver.Numerical_Solution_IDE import AdamOptimizerIDE
import matplotlib.pyplot as plt
import numpy as np

ADAM = True
ADAMIDE = True

np.random.seed(33)
theta = np.random.uniform(-10,10)

if ADAM:
    ###########
    ####ADAM###
    ###########

    optimizer = AdamOptimizer()
    theta_result = {}
    epochs = 2e3
    
    for epoch in range(int(epochs)):  
        grad = grad_f(theta)
        theta = optimizer.update(grad, theta)
        if epoch % 100 == 0:
            print(f"Iteration {epoch}: theta = {theta}")
        theta_result[epoch] = theta

    print(f"\nFinal parameter value ADAM: theta = {theta:.4f}")

    # Test the convergence of the Adam optimizer
    test_adam_convergence(theoretical_result=4.0)

    # Plot the parameter trajectory over time
    plt.figure(figsize=(10, 6))
    plt.plot(list(theta_result.keys()), list(theta_result.values()), label='Parameter trajectory ADAM')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title('Parameter trajectory over time')
    plt.legend()
    plt.grid(True)
    plt.show()

if ADAMIDE:
    ###################
    ##### ADAMIDE #####
    ###################

    t_max = 10
    t_span =(1e-12,t_max)
    optimizer = AdamOptimizerIDE(t_span=t_span, y0=[np.random.uniform(-10, 10)], verbose=True)
    sol = optimizer.optimize()

    plt.figure(figsize=(10, 5))
    plt.plot(sol.t, sol.y[0], label= 'Parameter trajectory ADAM IDE')
    plt.title('Solution for Theta(t) over Time')
    plt.xlabel('Time')
    plt.ylabel('Theta(t)')
    plt.grid(True)
    plt.legend()
    plt.show()
