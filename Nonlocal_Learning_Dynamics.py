from Optimizers.Adam import grad_f, AdamOptimizer, test_adam_convergence
from IDE_Solver.Numerical_Solution_IDE import AdamOptimizerIDE
import matplotlib.pyplot as plt
import numpy as np

ADAM = False
ADAMIDE = True

if ADAM:
    ###########
    ####ADAM###
    ###########

    optimizer = AdamOptimizer()
    theta = np.random.uniform(-1, 1)
    theta_result = {}
    epochs = 1e4

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

elif ADAMIDE:
    ###################
    ##### ADAMIDE #####
    ###################
    t_span =(1e-12,1e5)
    theta0 = [np.random.uniform(-1, 1)]
    optimizer = AdamOptimizerIDE(t_span=t_span, theta0=theta0)
    optimizer.optimize()

