import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sca(objective_function, lb, ub, dim, num_agents, max_iter):
    """
    Sine Cosine Algorithm (SCA) for optimization.
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

    return dest_pos, dest_fitness, convergence_curve

def shifted_sphere_f1(x):
    """
    Sphere function f1 with a shifted minimum based on Table A.1.
    """
    shift_vector = np.full_like(x, -30.0)
    return np.sum(np.square(x - shift_vector))

if __name__ == '__main__':
    # --- Parameters ---
    dim = 20
    lb = [-100] * dim
    ub = [100] * dim
    num_agents = 50
    max_iter = 1000
    THEORETICAL_F_MIN = 0

    # --- Run SCA ---
    best_solution, best_fitness, convergence_curve = sca(shifted_sphere_f1, lb, ub, dim, num_agents, max_iter)

    # --- Print Results ---
    print("\n------------------- RESULTS -------------------")
    print(f"Best solution found:\n{best_solution}")
    print(f"\nBest fitness value found: {best_fitness}")
    print(f"Theoretical f_min:      {THEORETICAL_F_MIN}")
    print(f"Difference (Error):       {abs(best_fitness - THEORETICAL_F_MIN)}")
    print("---------------------------------------------")

    # --- Plot 1: Convergence Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_curve)
    plt.title('Convergence Curve of SCA')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Value')
    plt.grid(True)
    plt.show()

    # --- Plot 2: Histogram of Best Solution Parameters ---
    plt.figure(figsize=(10, 6))
    plt.hist(best_solution, bins=15, color='skyblue', edgecolor='black')
    plt.title('Distribution of Parameters in the Best Solution')
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    
    # --- Prepare data for 2D/3D plots ---
    x_vals = np.linspace(best_solution[0] - 10, best_solution[0] + 10, 100)
    y_vals = np.linspace(best_solution[1] - 10, best_solution[1] + 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros(X.shape)
    fixed_params = best_solution[2:]
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            current_vec = np.concatenate(([X[i, j], Y[i, j]], fixed_params))
            Z[i, j] = shifted_sphere_f1(current_vec)
            
    # --- Plot 3: 2D Contour Plot ---
    plt.figure(figsize=(10, 8))
    cp = plt.contour(X, Y, Z, levels=20)
    plt.colorbar(cp, label='Fitness Value')
    plt.title('2D Fitness Landscape Contour (Slice)')
    plt.xlabel('Dimension 1 Value')
    plt.ylabel('Dimension 2 Value')
    plt.plot(best_solution[0], best_solution[1], 'r*', markersize=15, label='Best Solution Found')
    plt.legend()
    plt.show()

    # --- Plot 4: 3D Surface Plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.scatter(best_solution[0], best_solution[1], best_fitness, color='r', marker='*', s=200, label='Best Solution Found')
    ax.set_title('3D Fitness Landscape Surface (Slice)')
    ax.set_xlabel('Dimension 1 Value')
    ax.set_ylabel('Dimension 2 Value')
    ax.set_zlabel('Fitness Value')
    ax.legend()
    plt.show()