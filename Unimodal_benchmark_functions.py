import numpy as np
import math
import matplotlib.pyplot as plt

def sca(objective_function, lb, ub, dim, num_agents, max_iter):
    # init population
    positions = np.zeros((num_agents, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lb[i], ub[i], num_agents)

    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    # curves
    convergence_curve = np.zeros(max_iter)
    avg_fitness_curve = np.zeros(max_iter)
    x1_agent1_traj   = np.zeros(max_iter)

    # ✅ เก็บประวัติ (ฉาย 2 มิติแรก)
    search_history_xy = []  # list of (N_agents, 2) per iteration

    for t in range(max_iter):
        fitnesses = np.empty(num_agents)
        for i in range(num_agents):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitnesses[i] = objective_function(positions[i, :])

            if fitnesses[i] < dest_fitness:
                dest_fitness = fitnesses[i]
                dest_pos = positions[i, :].copy()

        convergence_curve[t] = dest_fitness
        avg_fitness_curve[t] = np.mean(fitnesses)

        # --- update (SCA)
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
        # ✅ บันทึกประวัติของ agent ทุกตัว (เฉพาะ 2D แรก)
        if dim >= 2:
            search_history_xy.append(positions[:, :2].copy())
        else:
            # ถ้า dim=1 ให้ขยายเป็น 2D โดยแกน y = 0
            search_history_xy.append(np.c_[positions[:, 0], np.zeros(num_agents)])

        if t % 100 == 0:
            print(f"Iteration {t}: Best Fitness = {dest_fitness}")

    # ---- plots: convergence + trajectory + average fitness (เหมือนเดิมย่อๆ) ----
    plt.figure(figsize=(8,5))
    plt.plot(convergence_curve, label="Best-so-far fitness"); plt.grid(True)
    plt.xlabel("Iterations"); plt.ylabel("Fitness"); plt.title("SCA Convergence"); plt.legend(); plt.show()

    plt.figure(figsize=(6,5))
    plt.plot(x1_agent1_traj, color='red', label="x₁ of agent #1"); plt.grid(True)
    plt.xlabel("Iterations"); plt.ylabel("x₁ value"); plt.title("Trajectory (agent #1)"); plt.legend(); plt.show()

    plt.figure(figsize=(6,5))
    plt.plot(avg_fitness_curve, color='green', label="Average fitness"); plt.grid(True)
    plt.xlabel("Iterations"); plt.ylabel("Average fitness"); plt.title("Average fitness of agents"); plt.legend(); plt.show()

    return dest_pos, dest_fitness, np.array(search_history_xy)

# ---------- ฟังก์ชันช่วย plot Search history (พร้อม contour) ----------
def plot_search_history(history_xy, lb, ub, contour_fn=None, optimum=None, title="Search history",
                        levels=20, sample_every=1):
    """
    history_xy: array shape (T, N, 2)
    contour_fn: ฟังก์ชัน f(x,y) -> z สำหรับวาด contour (ถ้า None จะวาดเฉพาะจุด)
    optimum: tuple/list (x*, y*) สำหรับแสดงดาวแดง
    """
    # พื้นที่พล็อต
    x_min, x_max = lb[0], ub[0]
    y_min, y_max = lb[1] if len(lb) > 1 else lb[0], ub[1] if len(ub) > 1 else ub[0]

    plt.figure(figsize=(6,6))

    # วาด contour ถ้ามี
    if contour_fn is not None:
        xs = np.linspace(x_min, x_max, 250)
        ys = np.linspace(y_min, y_max, 250)
        X, Y = np.meshgrid(xs, ys)
        Z = contour_fn(X, Y)
        plt.contour(X, Y, Z, levels=levels)

    # วาดจุดการค้นหา (ซ้ำซ้อนมากอาจ sample)
    T = history_xy.shape[0]
    for t in range(0, T, sample_every):
        pts = history_xy[t]
        plt.scatter(pts[:, 0], pts[:, 1], s=16, c='black')

    # optimum (ดาวแดง)
    if optimum is not None:
        plt.scatter([optimum[0]], [optimum[1]], s=120, c='red', marker='*')

    plt.xlim([x_min, x_max]); plt.ylim([y_min, y_max])
    plt.title(title); plt.xlabel("x₁"); plt.ylabel("x₂")
    plt.grid(False)
    plt.show()

# ---------------- example usage ----------------
if __name__ == '__main__':
    # ====== ตัวอย่างกับ Shifted Sphere: f(x)=sum((x+30)^2) ======
    def shifted_sphere_function(x):
        shift_vector = np.full_like(x, -30.0)
        return np.sum((x - shift_vector)**2)

    # contour 2D สำหรับรูป
    def sphere_contour_2d(X, Y, shift=(-30.0, -30.0)):
        x0, y0 = shift
        return (X - x0)**2 + (Y - y0)**2

    dim = 20            # ตั้ง 2 จะสวยสุด; >2 จะฉาย 2 มิติแรก
    lb = [-100] * dim
    ub = [100] * dim
    num_agents = 50
    max_iter = 1000

    best_solution, best_fitness, history_xy = sca(
        shifted_sphere_function, lb, ub, dim, num_agents, max_iter
    )

    print("\nBest solution found:\n", best_solution)
    print("\nBest fitness value:\n", best_fitness)

    # ====== พล็อต Search history แบบในรูป ======
    # ดาวแดงที่ optimum ของ shifted sphere คือ (-30, -30)
    plot_search_history(
        history_xy, lb, ub,
        contour_fn=sphere_contour_2d,      # ไม่มี contour ก็ใส่ None ได้
        optimum=(-30.0, -30.0),
        title="F1  Search history",
        levels=20,
        sample_every=5                      # วาดทุก 5 iterations เพื่อลดจุดทับกัน
    )
