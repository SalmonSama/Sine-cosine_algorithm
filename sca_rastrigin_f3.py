import numpy as np
import math
import matplotlib.pyplot as plt

# ================== SCA เดิม ==================
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
        # evaluate & update best
        for i in range(num_agents):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitness = objective_function(positions[i, :])
            if fitness < dest_fitness:
                dest_fitness = fitness
                dest_pos = positions[i, :].copy()

        # r1 decreases linearly 2 -> 0
        a = 2
        r1 = a - t * (a / max_iter)

        # update positions
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
# =================================================

# ---------- f3: Rastrigin ----------
def rastrigin_f3(x):
    A = 10.0
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))


# ---------- จำลองตำแหน่งเอเจนต์ ณ รอบที่ต้องการ ----------
def simulate_positions_until_iter(objective_function, lb, ub, dim, num_agents, iter_stop, seed=None):
    if seed is not None:
        np.random.seed(seed)

    positions = np.zeros((num_agents, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lb[i], ub[i], num_agents)

    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    for t in range(iter_stop + 1):
        # evaluate & update best
        for i in range(num_agents):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitness = objective_function(positions[i, :])
            if fitness < dest_fitness:
                dest_fitness = fitness
                dest_pos = positions[i, :].copy()

        if t == iter_stop:
            break

        # update positions
        a = 2
        r1 = a - t * (a / (iter_stop if iter_stop > 0 else 1))
        for i in range(num_agents):
            for j in range(dim):
                r2 = (2 * math.pi) * np.random.rand()
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + r1 * math.sin(r2) * abs(r3 * dest_pos[j] - positions[i, j])
                else:
                    positions[i, j] = positions[i, j] + r1 * math.cos(r2) * abs(r3 * dest_pos[j] - positions[i, j])

    return positions


# ---------- วาดกราฟ contourf + agents ----------
def plot_agents_on_rastrigin_easy(positions, lb, ub, iter_idx=0):
    xs = positions[:, 0]
    ys = positions[:, 1]

    n = 300
    x = np.linspace(lb[0], ub[0], n)
    y = np.linspace(lb[1], ub[1], n)
    X, Y = np.meshgrid(x, y)
    A = 10.0
    Z = A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.8)
    cs = ax.contour(X, Y, Z, levels=20, colors="black", linewidths=0.5, alpha=0.5)

    ax.scatter(xs, ys, s=50, c='red', edgecolors='white', linewidths=0.7, label="agents")
    ax.scatter([0], [0], s=180, marker='*', color='gold', edgecolors='black', label="global min (0,0)")

    ax.set_title(f"Agent positions at iter {iter_idx}", fontsize=14)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper right")
    ax.set_aspect('equal')

    plt.colorbar(cf, ax=ax, shrink=0.8, label="Fitness value")
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ================== main ==================
if __name__ == '__main__':
    N_RUNS = 20
    dim = 20
    lb = [-5] * dim
    ub = [5] * dim
    num_agents = 50
    max_iter = 800
    ITER_TO_PLOT = 3   # รอบที่จะพล็อตรูป

    plt.close('all')

    for run_idx in range(1, N_RUNS + 1):
        # พล็อต agent positions ที่ iter = ITER_TO_PLOT
        positions_now = simulate_positions_until_iter(
            rastrigin_f3, lb, ub, dim, num_agents, ITER_TO_PLOT, seed=run_idx
        )
        plot_agents_on_rastrigin_easy(positions_now, lb, ub, iter_idx=ITER_TO_PLOT)

        # รัน SCA เดิม (เพื่อได้ผลลัพธ์จริง ๆ)
        np.random.seed(run_idx)
        best_solution, best_fitness = sca(rastrigin_f3, lb, ub, dim, num_agents, max_iter)
        print(f"[Run {run_idx:02d}] Best fitness = {best_fitness:.6g}")