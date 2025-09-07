import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # OK ถ้าไม่ใช้ก็ลบได้

# ---------------- SCA core ----------------
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
    search_history = []  # store first-2D projection over time

    # Main loop
    for t in range(max_iter):
        agent_fitnesses = np.zeros(num_agents)

        # Evaluate the fitness of each agent
        for i in range(num_agents):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitness = objective_function(positions[i, :])
            agent_fitnesses[i] = fitness

            # Update global best
            if fitness < dest_fitness:
                dest_fitness = fitness
                dest_pos = positions[i, :].copy()

        # Update r1 (exploration -> exploitation)
        a = 2.0
        r1 = a - t * (a / max_iter)

        # Update the position of each agent
        for i in range(num_agents):
            for j in range(dim):
                r2 = (2 * math.pi) * np.random.rand()
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()
                diff = abs(r3 * dest_pos[j] - positions[i, j])

                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + (r1 * math.sin(r2) * diff)
                else:
                    positions[i, j] = positions[i, j] + (r1 * math.cos(r2) * diff)

        # ----- Store data for plotting -----
        convergence_curve[t] = dest_fitness
        avg_fitness_history[t] = np.mean(agent_fitnesses)
        agent1_x1_trajectory[t] = positions[0, 0]
        search_history.append(positions[:, :2].copy())

    return dest_pos, dest_fitness, convergence_curve, avg_fitness_history, agent1_x1_trajectory, search_history


# ---------------- Benchmarks ----------------
def shifted_sphere_f1(x):
    """
    Shifted Sphere (Table A.1 style): optimum at x = [-30, -30, ..., -30], f_min = 0
    """
    shift_vector = np.full_like(x, -30.0)
    return np.sum((x - shift_vector) ** 2)


# ---------------- Run & Plot ----------------
if __name__ == '__main__':
    # เคลียร์ figure เก่าใน Spyder ป้องกันเตือนหลายรูปค้าง
    plt.close('all')

    # --- Parameters ---
    dim = 20
    lb = [-100.0] * dim
    ub = [100.0] * dim

    # เลือกพารามิเตอร์ตามสไตล์ใน PDF (30 เอเจนต์ / 500 รอบ) — ปรับได้ตามต้องการ
    num_agents = 30
    max_iter = 500

    # ค่าทฤษฎีของ Shifted Sphere
    THEORETICAL_F_MIN = 0.0
    THEORETICAL_OPTIMUM = np.full(dim, -30.0)

    # --- Run SCA ---
    best_solution, best_fitness, convergence, avg_fit, traj_x1, hist_xy = sca(
        shifted_sphere_f1, lb, ub, dim, num_agents, max_iter
    )

    # --- Print Results ---
    print("\n------------------- RESULTS -------------------")
    print(f"Best solution found:\n{best_solution}")
    print(f"\nBest fitness value found: {best_fitness}")
    print(f"Theoretical f_min:        {THEORETICAL_F_MIN}")
    print(f"Absolute error:           {abs(best_fitness - THEORETICAL_F_MIN)}")
    print("-----------------------------------------------")

    # ---- Plot 1: Convergence Curve ----
    plt.figure(figsize=(10, 6))
    plt.plot(convergence, label="Best-so-far fitness")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("SCA Convergence on Shifted Sphere (F1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 2: Trajectory of x1 (agent #1) ----
    plt.figure(figsize=(10, 6))
    plt.plot(traj_x1, label="x₁ of agent #1")
    plt.axhline(y=THEORETICAL_OPTIMUM[0], linestyle='--',
                label=f'Optimum x₁ = {THEORETICAL_OPTIMUM[0]}')
    plt.xlabel("Iterations")
    plt.ylabel("x₁ value")
    plt.title("Trajectory of the first variable (agent #1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 3: Average fitness of all agents ----
    plt.figure(figsize=(10, 6))
    plt.plot(avg_fit, label="Average fitness of agents")
    plt.xlabel("Iterations")
    plt.ylabel("Average fitness")
    plt.title("Average fitness during optimization on F1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 4: Search history (2D) + Contour background ----
    # ทำพื้นหลังคอนทัวร์เร็ว ๆ (รูปปั้น 2D ของ shifted sphere โดยตรึงมิติอื่นไว้ที่ -30)
    plt.figure(figsize=(10, 8))
    x_range = np.linspace(lb[0], ub[0], 200)
    y_range = np.linspace(lb[1], ub[1], 200)
    X, Y = np.meshgrid(x_range, y_range)

    # สำหรับ shifted sphere: f(x) = Σ (x_i + 30)^2
    # ถ้าตรึงมิติที่ 3..D = -30, ค่าคงที่ของมิติอื่นจะเป็น 0
    Z = (X + 30.0) ** 2 + (Y + 30.0) ** 2

    cp = plt.contour(X, Y, Z, levels=25)
    plt.colorbar(cp, label='Fitness (partial 2D view)')

    sample_every = 5
    for i, pts in enumerate(hist_xy):
        if i % sample_every == 0:
            alpha = (i / max_iter) * 0.7 + 0.1
            plt.scatter(pts[:, 0], pts[:, 1], s=10, alpha=alpha)

    plt.plot(THEORETICAL_OPTIMUM[0], THEORETICAL_OPTIMUM[1],
             marker='P', markersize=12, linestyle='None', label='Theoretical Optimum')
    plt.plot(best_solution[0], best_solution[1],
             marker='*', markersize=14, linestyle='None', label='Best Solution Found')
    plt.title("Search History on F1 (2D Projection)")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot 5: 3D surface around best solution ----
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    span = 50.0
    x_vals = np.linspace(best_solution[0] - span, best_solution[0] + span, 120)
    y_vals = np.linspace(best_solution[1] - span, best_solution[1] + span, 120)
    Xs, Ys = np.meshgrid(x_vals, y_vals)

    # คงมิติที่ 3..D ไว้ที่ค่า best_solution เพื่อดูบริเวณรอบคำตอบ
    # f(x) = (x1+30)^2 + (x2+30)^2 + Σ_{k=3..D} (best_k + 30)^2  (ค่าหลังเป็นค่าคงที่)
    const_rest = np.sum((best_solution[2:] + 30.0) ** 2)
    Zs = (Xs + 30.0) ** 2 + (Ys + 30.0) ** 2 + const_rest

    surf = ax.plot_surface(Xs, Ys, Zs, edgecolor='none', alpha=0.85)
    zmin = float(Zs.min())
    ax.contour(Xs, Ys, Zs, zdir='z', offset=zmin, levels=25)

    ax.scatter(best_solution[0], best_solution[1], best_fitness,
               marker='*', s=200, label='Best Solution Found')

    ax.set_title("Shifted Sphere F1 (optimum at -30)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("fitness")
    ax.view_init(elev=35, azim=-60)
    ax.legend()
    plt.tight_layout()
    plt.show()
