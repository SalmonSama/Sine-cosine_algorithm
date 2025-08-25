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
    # MODIFIED: New objective function F14 (Composite Function 1)
    def composite_function_f14(x):
        """
        Composite Function F14 (CF1) from the benchmark suite.
        It is a composition of 10 shifted Sphere functions.
        """
        dim = 10
        num_functions = 10
        
        # Standard optima (shift positions) for 10D CF1
        optima = np.array([
            [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
            [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
        ])
        
        lambdas = np.full(num_functions, 5.0 / 100.0)
        sigmas = np.full(num_functions, 1.0)
        biases = np.zeros(num_functions) # Bias for CF1 is 0

        # Basic Sphere function
        def sphere(z):
            return np.sum(np.square(z))

        # 1. Calculate weights (w_i)
        weights = np.zeros(num_functions)
        for i in range(num_functions):
            dist_sq = np.sum(np.square(x - optima[i]))
            if dist_sq != 0:
                weights[i] = np.exp(-dist_sq / (2 * dim * sigmas[i]**2))
            else:
                weights[i] = 1e10 # Give a very high weight if at an optimum
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        # 2. Calculate final fitness
        total_fitness = 0
        for i in range(num_functions):
            z = (x - optima[i]) / lambdas[i]
            f_i = sphere(z)
            total_fitness += weights[i] * (f_i + biases[i])
            
        return total_fitness

    # MODIFIED: Parameters updated according to the image for F14
    dim = 10
    lb = [-5] * dim  # Range is now [-5, 5]
    ub = [5] * dim   # Range is now [-5, 5]
    num_agents = 50
    max_iter = 1000

    # ADDED: Define the theoretical f_min for comparison.
    THEORETICAL_F_MIN = 0

    # MODIFIED: Changed the function call to use the new F14 function
    best_solution, best_fitness = sca(composite_function_f14, lb, ub, dim, num_agents, max_iter)

    print("\n------------------- RESULTS -------------------")
    print(f"Best solution found:\n{best_solution}")
    print(f"\nBest fitness value found: {best_fitness}")
    print(f"Theoretical f_min:      {THEORETICAL_F_MIN}")
    print(f"Difference (Error):       {abs(best_fitness - THEORETICAL_F_MIN)}")
    print("---------------------------------------------")