from optimizer.adam import grad_f, Adam, mse_loss, grad_mse
from ide_solver.ide import AdamIDE
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time

ADAM = True
ADAMIDE = False
example = 1

param_grid = {
    'lr': [0.1, 0.01, 0.001],
    'beta1': [0.9, 0.25],
    'beta2': [0.999, 0.25],
}

# Generar todas las combinaciones posibles de parámetros
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

np.random.seed(33)
theta_initial = np.random.uniform(-10,10)
epochs = int(1e4)

if example == 2:
    x = np.arange(1,100+1, 1)
    y_true = 2*x # True values for y = 2x
else:
    y_true = None
    x = None

fig, axs = plt.subplots(ncols=3, figsize=(10, 15), sharex=True, sharey=True)

if ADAM:
    ###########
    ####ADAM###
    ###########

    fig.suptitle('Convergence Trajectories for different Adam Configurations')

    for i, lr in enumerate(param_grid['lr']):
        ax = axs[i]

        for params in [p for p in all_params if p['lr'] == lr]:

            start_time =time.time()

            print(f'Adam Configuration: {params}')
            optimizer = Adam(**params)
            theta = theta_initial
            theta_result = []
            losses =[]
                    
            for epoch in range(epochs):
                if example == 1:  
                    grad = grad_f(theta)
                elif example == 2:
                    y_pred = x * theta
                    loss = mse_loss(y_true, y_pred)
                    grad = grad_mse(theta, x, y_true)
                    losses.append(loss)

                theta = optimizer.update(grad, theta)
                theta_result.append(theta)
                if epoch % 100 == 0:
                    print(f"Iteration {epoch}: theta = {theta}")
                
            label = f"beta1={params['beta1']}, beta2={params['beta2']}"
            if example == 1:
                ax.plot(theta_result, label=label)
            elif example == 2:
                ax.plot(losses, label=label)


            elapsed_time = (time.time() - start_time)/60
            print(f"\nFinal parameter value ADAM: theta = {theta:.4f}, with elapsed time: {elapsed_time:.2f} minutes.")

        ax.set_title(f'Learning Rate = {lr}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Theta value' if example == 1 else 'Loss')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if ADAMIDE:
    ###################
    ##### ADAMIDE #####
    ###################

    fig.suptitle('Convergence Trajectories for Different first-order nonlocal continuous Adam Configurations')

    t_max = 15
    t_span =(1e-12,t_max)
    y0 = [np.random.uniform(-10, 10)]

    for i, lr in enumerate(param_grid['lr']):
        ax = axs[i]
        filtered_params = [p for p in all_params if p['lr'] == lr]

        for params in filtered_params:

            start_time = time.time()

            print(f'\nNonlocal continuous Adam Configuration: {params}')

            optimizer = AdamIDE(t_span=t_span, alpha=lr, beta=[params['beta1'], params['beta2']], y0=y0, example=example, y_true=y_true, x=x, verbose=False)
            
            if example == 1:
                sol = optimizer.optimize()
            elif example == 2:
                sol, losses_IDE = optimizer.optimize_losses()

            elapsed_time = (time.time() - start_time)/60

            print(f"Final parameter value ADAMIDE: theta = {sol.y[0][-1]:.4f}, with elapsed time: {elapsed_time:.2f} minutes.")

            label = f"beta1={params['beta1']}, beta2={params['beta2']}"
            if example == 1:
                ax.plot(sol.t, sol.y[0], label=label)
                ax.set_ylabel('Theta value')
            elif example == 2:
                ax.plot(losses_IDE['Time'], losses_IDE['Loss'], label=label)
                ax.set_ylabel('Loss')
            ax.set_title(f'Learning Rate = {lr}')
            ax.set_yscale('log' if example == 2 else 'linear')
            ax.set_xlabel('Time')
            ax.legend()
            ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()