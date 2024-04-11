from Optimizers.Adam import grad_f, adam_optimization
import matplotlib.pyplot as plt

# Define the initial parameter value and the number of iterations
theta = 0.0
num_iterations = 1000

theta_t = {}

for iteration in range(num_iterations):

    theta_t[iteration] = theta

    # Compute the gradient of the loss function with respect to theta
    grad = grad_f(theta)
    
    # Update the parameter using the Adam optimization algorithm
    theta, m, v = adam_optimization(grad, theta, t=iteration+1, m=0.0, v=0.0)
    
    # Print the current parameter value every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: theta = {theta:.4f}")

print(f"Final parameter value: theta = {theta:.4f}")

# Plot the parameter trajectory over time
plt.figure(figsize=(10, 6))
plt.plot(list(theta_t.keys()), list(theta_t.values()), label='Parameter trajectory')
plt.xlabel('Iteration')
plt.ylabel('Parameter value')
plt.title('Parameter trajectory over time')
plt.legend()
plt.grid(True)
plt.show()

