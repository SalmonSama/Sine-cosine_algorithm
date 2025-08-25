import numpy as np
import math
import matplotlib.pyplot as plt

# ---------- SCA ----------
def sca(objective_function, lb, ub, dim, num_agents, max_iter):
    positions = np.zeros((num_agents, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lb[i], ub[i], num_agents)

    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    # curves
    convergence_curve = np.zeros(max_iter)
    avg_fitness_curve = np.zeros(max_iter)
    x1_agent1_traj = np.zeros(max_iter)

    # Store history (first 2D projection for the Search history graph)
    search_history_xy = []

    for t in range(max_iter):
        # ---- evaluate once per iter
        fitnesses = np.empty(num_agents)
        for i in range(num_agents):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitnesses[i] = objective_function(positions[i, :])
            if fitnesses[i] < dest_fitness:
                dest_fitness = fitnesses[i]
                dest_pos = positions[i, :].copy()

        convergence_curve[t] = dest_fitness
        avg_fitness_curve[t] = np.mean(fitnesses)

        # ---- update (SCA move)
        a = 2
        r1 = a - t * (a / max_iter)
        for i in range(num_agents):
            for j in range(dim):
                r2 = (2 * math.pi) * np.random.rand()
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + r1 * math.sin(r2) * abs(r3 * dest_pos[j] - positions[i, j])
                else:
                    positions[i, j] = positions[i, j] + r1 * math.cos(r2) * abs(r3 * dest_pos[j] - positions[i, j])

        x1_agent1_traj[t] = positions[0, 0]

        # Store history of the first 2D dimensions
        if dim >= 2:
            search_history_xy.append(positions[:, :2].copy())
        else:
            search_history_xy.append(np.c_[positions[:, 0], np.zeros(num_agents)])

        if t % 100 == 0:
            print(f"Iteration {t}: Best Fitness = {dest_fitness}")

    return dest_pos, dest_fitness, convergence_curve, avg_fitness_curve, x1_agent1_traj, np.array(search_history_xy)

# ---------- Helper function to plot Search history ----------
def plot_search_history(history_xy, lb, ub, contour_fn=None, optimum=None,
                        title="Search history", levels=20, sample_every=5):
    x_min, x_max = lb[0], ub[0]
    y_min = lb[1] if len(lb) > 1 else lb[0]
    y_max = ub[1] if len(ub) > 1 else ub[0]

    plt.figure(figsize=(6,6))

    if contour_fn is not None:
        xs = np.linspace(x_min, x_max, 250)
        ys = np.linspace(y_min, y_max, 250)
        X, Y = np.meshgrid(xs, ys)
        Z = contour_fn(X, Y)
        plt.contour(X, Y, Z, levels=levels)

    T = history_xy.shape[0]
    for t in range(0, T, sample_every):
        pts = history_xy[t]
        plt.scatter(pts[:, 0], pts[:, 1], s=16, c='black')

    if optimum is not None:
        plt.scatter([optimum[0]], [optimum[1]], s=120, c='red', marker='*')

    plt.xlim([x_min, x_max]); plt.ylim([y_min, y_max])
    plt.title(title); plt.xlabel("x₁"); plt.ylabel("x₂")
    plt.grid(False)
    plt.show()

# ---------- F8 Schwefel (shifted) ----------
def shifted_schwefel_f8(x):
    """
    F8 Schwefel with shift o = [-300,...,-300]
    f(x) = sum( - (x - o)_i * sin( sqrt(|(x - o)_i|) ) )
    """
    shift_vector = np.full_like(x, -300.0)  # o
    shifted_x = x - shift_vector            # = x + 300
    return np.sum(-shifted_x * np.sin(np.sqrt(np.abs(shifted_x))))

# Contour function for F8 in 2D (uses the same formula)
def schwefel_contour_2d(X, Y, shift=(-300.0, -300.0)):
    sx = X - shift[0]
    sy = Y - shift[1]
    return -(sx * np.sin(np.sqrt(np.abs(sx)))) - (sy * np.sin(np.sqrt(np.abs(sy))))

# ------------------- Run and Plot -------------------
if __name__ == '__main__':
    dim = 20
    lb = [-500] * dim
    ub = [500] * dim
    num_agents = 50
    max_iter = 1000

    THEORETICAL_F_MIN = dim * -418.9829
    # Optimum position of standard Schwefel is at ~420.9687 per dimension
    # When shifted by o=-300, the new optimum x* is ≈ -300 + 420.9687 = 120.9687
    OPT_2D = (120.9687, 120.9687)

    best_solution, best_fitness, conv, avg_fit, traj_x1, hist_xy = sca(
        shifted_schwefel_f8, lb, ub, dim, num_agents, max_iter
    )

    print("\n------------------- RESULTS -------------------")
    print("Best solution found:\n", best_solution)
    print("\nBest fitness value found:", best_fitness)
    print("Theoretical f_min:       ", THEORETICAL_F_MIN)
    print("Difference (Error):      ", abs(best_fitness - THEORETICAL_F_MIN))
    print("---------------------------------------------")

    # 1) Convergence
    plt.figure(figsize=(8,5))
    plt.plot(conv, label="Best-so-far fitness")
    plt.xlabel("Iterations"); plt.ylabel("Fitness")
    plt.title("SCA Convergence on F8 (shifted Schwefel)")
    plt.grid(True); plt.legend(); plt.show()

    # 2) Trajectory of x1 (agent #1)
    plt.figure(figsize=(6,5))
    plt.plot(traj_x1, color='red', label="x₁ of agent #1")
    plt.xlabel("Iterations"); plt.ylabel("x₁ value")
    plt.title("Trajectory of the first variable (agent #1)")
    plt.grid(True); plt.legend(); plt.show()

    # 3) Average fitness of all agents
    plt.figure(figsize=(6,5))
    plt.plot(avg_fit, color='green', label="Average fitness of agents")
    plt.xlabel("Iterations"); plt.ylabel("Average fitness")
    plt.title("Average fitness during optimization (F8)")
    plt.grid(True); plt.legend(); plt.show()

    # 4) Search history (first 2D projection)
    plot_search_history(
        hist_xy, lb, ub,
        contour_fn=schwefel_contour_2d,
        optimum=OPT_2D,
        title="F8 Search history",
        levels=25,
        sample_every=5
    )