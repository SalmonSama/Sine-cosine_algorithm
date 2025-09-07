# run_f3_20_images.py — ใช้ SCA + f3 (Rastrigin) รัน 20 รอบ เซฟ "รูป 3D surface" 20 รูป
import os
import math
import numpy as np

# ====== ฟังก์ชันเดิมของคุณ (คงเดิม ไม่แก้) ======
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
# ================================================

# ---------- f3: Rastrigin ----------
def rastrigin_f3(x):
    A = 10.0
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))

if __name__ == '__main__':
    import matplotlib
    # ถ้าไม่มีจอ (เช่น WSL/เซิร์ฟเวอร์) ให้เซฟรูปอย่างเดียว
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    # ====== ตั้งค่าหลัก ======
    N_RUNS = 20          # จำนวนรูป (1 รอบ = 1 รูป)
    dim = 20             # จะใช้ 20 เหมือนโค้ดคุณ; ถ้าอยากให้กราฟสื่อความง่ายสุด ตั้งเป็น 2
    lb = [-5] * dim
    ub = [ 5] * dim
    num_agents = 50
    max_iter = 800
    THEORETICAL_F_MIN = 0.0

    # โฟลเดอร์ผลลัพธ์
    outdir = Path("f3_outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = outdir / f"batch_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # ====== รัน 20 รอบ และเซฟ "รูปพื้นผิว 3D" 20 รูป ======
    # ใช้ seed ต่างกันเพื่อให้ผลต่างกันต่อรูป
    for run_idx in range(1, N_RUNS + 1):
        np.random.seed(run_idx)
        best_solution, best_fitness = sca(rastrigin_f3, lb, ub, dim, num_agents, max_iter)

        print(f"[Run {run_idx:02d}] best_fitness = {best_fitness:.6g}")

        # -------- พล็อตพื้นผิว Rastrigin (ใช้แค่ 2 มิติแรกสำหรับวาดรูป) --------
        n = 200
        xg = np.linspace(lb[0], ub[0], n)
        yg = np.linspace(lb[1], ub[1], n)
        X, Y = np.meshgrid(xg, yg)
        A = 10.0
        Z = A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

        fig = plt.figure(figsize=(6.8, 4.8), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        zmin = float(Z.min())
        ax.contour(X, Y, Z, zdir="z", offset=zmin, levels=25)

        # จุดคำตอบที่ดีที่สุด (ใช้ 2 มิติแรกสำหรับวางบนรูป)
        bx, by = best_solution[0], best_solution[1]
        bz = A * 2 + (bx**2 - A * np.cos(2 * np.pi * bx)) + (by**2 - A * np.cos(2 * np.pi * by))
        ax.scatter([bx], [by], [bz], marker="x", s=80)

        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("fitness value")
        ax.set_title(f"Rastrigin f3 (Run {run_idx:02d}) | best={best_fitness:.3g}")
        ax.view_init(elev=35, azim=-60)

        out_img = outdir / f"run_{run_idx:02d}_surface.png"
        plt.tight_layout()
        plt.savefig(out_img, dpi=180)
        plt.close()

    print(f"\nSaved 20 images to: {outdir.resolve()}\n")

    # ====== (ทางเลือก) อยากได้ 'Convergence' 20 รูป? เปิดคอมเมนต์ด้านล่างแทนได้ ======
    # # วาด Convergence แยกรอบ (จะได้ 20 รูปเหมือนกัน แต่เป็นกราฟเส้น)
    # outdir2 = outdir / "convergence_only"
    # outdir2.mkdir(parents=True, exist_ok=True)
    # for run_idx in range(1, N_RUNS + 1):
    #     np.random.seed(run_idx)
    #     # เก็บ history เองเล็กน้อย: ใช้ SCA เดิม + ดัดส่วนในลูปเพื่อเก็บค่า
    #     # ทางง่าย: clone SCA มาแก้นิดหน่อย แต่คุณขอ "ใช้ฟังก์ชันเดิม" เลยขอข้าม
    #     # ทางเลือก: เรียกซ้ำ SCA ทีละช่วงแล้ว append history (ซับซ้อน)
    #     # แนะนำให้ใช้ SCA เวอร์ชันที่ return history จะสะดวกกว่า
    #     pass
