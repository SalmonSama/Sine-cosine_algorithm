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

    # ----- Data for plotting -----
    convergence_curve = np.zeros(max_iter)
    avg_fitness_history = np.zeros(max_iter)
    agent1_x1_trajectory = np.zeros(max_iter)
    search_history = [] # To store 2D projection of agents' positions

    # Main loop
    for t in range(max_iter):
        agent_fitnesses = np.zeros(num_agents)
        # Evaluate the fitness of each agent
        for i in range(num_agents):
            # Check if the positions are within the bounds
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            
            fitness = objective_function(positions[i, :])
            agent_fitnesses[i] = fitness

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
        
        # ----- Store data for plotting -----
        convergence_curve[t] = dest_fitness
        avg_fitness_history[t] = np.mean(agent_fitnesses)
        agent1_x1_trajectory[t] = positions[0, 0] # Trajectory of the first variable of the first agent
        search_history.append(positions[:, :2].copy()) # Store the first two dimensions

    return dest_pos, dest_fitness, convergence_curve, avg_fitness_history, agent1_x1_trajectory, search_history

def shifted_sphere_f1(x):
    """
    Sphere function f1 with a shifted minimum based on Table A.1.
    Theoretical minimum is at x = [-30, -30, ..., -30].
    RMSE version to reduce scale of fitness values.
    """
    shift_vector = np.full_like(x, -30.0)
    return np.sqrt(np.mean(np.square(x - shift_vector)))

if __name__ == '__main__':
    # --- Parameters ---
    dim = 20
    lb = [-100] * dim
    ub = [100] * dim
    num_agents = 200
    max_iter = 1000
    THEORETICAL_F_MIN = 0
    THEORETICAL_OPTIMUM = np.full(dim, -30.0)

    # --- Run SCA ---
    best_solution, best_fitness, convergence, avg_fit, traj_x1, hist_xy = sca(
        shifted_sphere_f1, lb, ub, dim, num_agents, max_iter
    )

    # --- Print Results ---
    print("\n------------------- RESULTS -------------------")
    print(f"Best solution found:\n{best_solution}")
    print(f"\nBest fitness value found: {best_fitness}")
    print(f"Theoretical f_min:      {THEORETICAL_F_MIN}")
    print(f"Difference (Error):     {abs(best_fitness - THEORETICAL_F_MIN)}")
    print("---------------------------------------------")
    
    # --- Plotting Section (same as before) ---
    # Plot 1: Convergence Curve
    plt.figure(figsize=(10, 6))
    plt.plot(convergence, label="Best-so-far fitness")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness (RMSE)")
    plt.title("SCA Convergence on Shifted Sphere (F1, RMSE)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 2: Trajectory of x1 (agent #1)
    plt.figure(figsize=(10, 6))
    plt.plot(traj_x1, color='red', label="x₁ of agent #1")
    plt.xlabel("Iterations")
    plt.ylabel("x₁ value")
    plt.title("Trajectory of the first variable (agent #1)")
    plt.axhline(y=THEORETICAL_OPTIMUM[0], color='black', linestyle='--', label=f'Theoretical Optimum ({THEORETICAL_OPTIMUM[0]})')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 3: Average fitness of all agents
    plt.figure(figsize=(10, 6))
    plt.plot(avg_fit, color='green', label="Average fitness of agents")
    plt.xlabel("Iterations")
    plt.ylabel("Average RMSE fitness")
    plt.title("Average fitness during optimization on F1 (RMSE)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 4: Search history (first 2D projection)
    plt.figure(figsize=(10, 8))
    x_range = np.linspace(lb[0], ub[0], 100)
    y_range = np.linspace(lb[1], ub[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    fixed_params = np.full(dim - 2, -30.0)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = shifted_sphere_f1(np.concatenate(([X[i, j], Y[i, j]], fixed_params)))
    cp = plt.contour(X, Y, Z, levels=25)
    plt.colorbar(cp, label='Fitness (RMSE)')
    sample_every = 5
    for i, positions in enumerate(hist_xy):
        if i % sample_every == 0:
            alpha = (i / max_iter) * 0.7 + 0.1
            plt.scatter(positions[:, 0], positions[:, 1], color='blue', s=10, alpha=alpha)
    plt.plot(THEORETICAL_OPTIMUM[0], THEORETICAL_OPTIMUM[1], 'gP', markersize=15, label='Theoretical Optimum')
    plt.plot(best_solution[0], best_solution[1], 'r*', markersize=15, label='Best Solution Found')
    plt.title("Search History on F1 (2D Projection)")
    plt.xlabel("x₁ value")
    plt.ylabel("x₂ value")
    plt.legend()
    plt.show()

    # Plot 5: 3D surface plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    x_vals = np.linspace(best_solution[0] - 50, best_solution[0] + 50, 100)
    y_vals = np.linspace(best_solution[1] - 50, best_solution[1] + 50, 100)
    X_surf, Y_surf = np.meshgrid(x_vals, y_vals)
    Z_surf = np.zeros(X_surf.shape)
    for i in range(X_surf.shape[0]):
        for j in range(X_surf.shape[1]):
            current_vec = np.concatenate(([X_surf[i, j], Y_surf[i, j]], best_solution[2:]))
            Z_surf[i, j] = shifted_sphere_f1(current_vec)
    ax.plot_surface(X_surf, Y_surf, Z_surf, cmap='viridis', edgecolor='none', alpha=0.8)
    zmin = float(Z_surf.min())
    ax.contour(X_surf, Y_surf, Z_surf, zdir='z', offset=zmin, levels=25, cmap='viridis')
    ax.scatter(best_solution[0], best_solution[1], best_fitness, color='r', marker='*', s=200, label='Best Solution Found')
    ax.set_title("Shifted Sphere F1 (o=-30, RMSE)")
    ax.set_xlabel("x values")
    ax.set_ylabel("y values")
    ax.set_zlabel("Fitness (RMSE)")
    ax.view_init(elev=35, azim=-60)
    ax.legend()
    plt.show()