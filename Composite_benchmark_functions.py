import numpy as np
import math

def sca(objective_function, lb, ub, dim, num_agents, max_iter):
    """
    Sine Cosine Algorithm (SCA) for optimization.
    """
    positions = np.zeros((num_agents, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lb[i], ub[i], num_agents)

    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")
    convergence_curve = np.zeros(max_iter)

    for t in range(max_iter):
        for i in range(num_agents):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitness = objective_function(positions[i, :])
            if fitness < dest_fitness:
                dest_fitness = fitness
                dest_pos = positions[i, :].copy()

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

        convergence_curve[t] = dest_fitness
        if (t % 100 == 0):
            print(f"Iteration {t}: Best Fitness = {dest_fitness}")

    return dest_pos, dest_fitness


if __name__ == '__main__':
    # ---------- Objective function: f1 (Shifted Sphere) ----------
    def shifted_sphere_f1(x):
        shift_vector = np.full_like(x, -30.0)  # o = [-30,...]
        return np.sum(np.square(x - shift_vector))

    dim = 20
    lb = [-100] * dim
    ub = [100] * dim
    num_agents = 50
    max_iter = 1000
    THEORETICAL_F_MIN = 0

    best_solution, best_fitness = sca(shifted_sphere_f1, lb, ub, dim, num_agents, max_iter)

    print("\n------------------- RESULTS -------------------")
    print(f"Best solution found:\n{best_solution}")
    print(f"\nBest fitness value found: {best_fitness}")
    print(f"Theoretical f_min:      {THEORETICAL_F_MIN}")
    print(f"Difference (Error):       {abs(best_fitness - THEORETICAL_F_MIN)}")
    print("---------------------------------------------")

    # ---------- 3D Surface Plot ----------
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = np.linspace(-60, 0, 200)
    y = np.linspace(-60, 0, 200)
    X, Y = np.meshgrid(x, y)
    Z = (X + 30.0)**2 + (Y + 30.0)**2

    fig = plt.figure(figsize=(6.4, 4.6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    zmin = float(Z.min())
    ax.contour(X, Y, Z, zdir="z", offset=zmin, levels=20)
    ax.set_xlabel("x values"); ax.set_ylabel("y values"); ax.set_zlabel("fitness value")
    ax.set_title("Shifted Sphere f1 (o=-30)")
    ax.view_init(elev=35, azim=-60)
    plt.show()
