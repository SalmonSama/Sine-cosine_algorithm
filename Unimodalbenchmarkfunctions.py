import numpy as np
import math

def sca(objective_function, lb, ub, dim, num_agents, max_iter):
    """
    Sine Cosine Algorithm (SCA) for optimization.

    Args:
        objective_function (function): The objective function to be minimized.
        lb (list or np.array): The lower bounds of the search space.
        ub (list or np.array): The upper bounds of the search space.
        dim (int): The dimension of the search space.
        num_agents (int): The number of search agents (population size).
        max_iter (int): The maximum number of iterations.

    Returns:
        tuple: A tuple containing the best solution found and its corresponding fitness value.
    """
    # Initialize the positions of the search agents
    positions = np.zeros((num_agents, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lb[i], ub[i], num_agents)

    # Initialize the destination position and fitness
    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    # Convergence curve
    convergence_curve = np.zeros(max_iter)

    # Main loop
    for t in range(max_iter):
        # Evaluate the fitness of each agent
        for i in range(num_agents):
            # Check if the positions are within the bounds
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            
            fitness = objective_function(positions[i, :])

            # Update the destination if a better solution is found
            if fitness < dest_fitness:
                dest_fitness = fitness
                dest_pos = positions[i, :].copy()

        # Update r1 (the exploration/exploitation parameter)
        a = 2
        r1 = a - t * (a / max_iter)

        # Update the position of each agent
        for i in range(num_agents):
            for j in range(dim):
                # Update r2, r3, and r4
                r2 = (2 * math.pi) * np.random.rand()
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()

                # Update the position based on the sine or cosine function
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + (r1 * math.sin(r2) * abs(r3 * dest_pos[j] - positions[i, j]))
                else:
                    positions[i, j] = positions[i, j] + (r1 * math.cos(r2) * abs(r3 * dest_pos[j] - positions[i, j]))
        
        convergence_curve[t] = dest_fitness
        if (t % 100 == 0):
            print(f"Iteration {t}: Best Fitness = {dest_fitness}")

    return dest_pos, dest_fitness

if __name__ == '__main__':
    # MODIFIED: Changed the function to include the shift position
    def shifted_sphere_function(x):
        """
        Sphere function with a shifted minimum based on the provided image.
        f(x) = sum((x_i - o_i)^2)
        """
        # Create the shift vector o = [-30, -30, ..., -30]
        shift_vector = np.full_like(x, -30.0)
        
        # Calculate the sum of squares for the shifted vector
        return np.sum(np.square(x - shift_vector))

    # Parameters from the image and the original file
    dim = 20
    lb = [-100] * dim
    ub = [100] * dim
    num_agents = 50
    max_iter = 1000

    # MODIFIED: Changed the function call to use the new shifted function
    best_solution, best_fitness = sca(shifted_sphere_function, lb, ub, dim, num_agents, max_iter)

    print("\nBest solution found:")
    print(best_solution)
    print("\nBest fitness value:")
    print(best_fitness)