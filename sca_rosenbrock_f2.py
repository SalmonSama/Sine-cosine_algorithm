# -*- coding: utf-8 -*-
"""
SCA on Rosenbrock (f2) — D=2 visuals
------------------------------------
Generates:
1) Contour plot (D=2)  -> contour_f2_rosenbrock.png
2) Agent slideshow (first 20 iterations, 20 agents) -> rosenbrock_agents/iter_XX.png + GIF
3) Convergence plot (iteration 5..100) -> rosenbrock_convergence.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# Benchmark: Rosenbrock f2
# =========================
def rosenbrock_f2(x: np.ndarray) -> float:
    """
    f2(x1, x2) = 100*(x2 - x1**2)**2 + (1 - x1)**2
    Domain: [-2, 2]^2
    Global minimum at (1,1) with f=0
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x1, x2 = x
    return float(100.0 * (x2 - x1**2)**2 + (1.0 - x1)**2)


# =========================
# Sine Cosine Algorithm
# =========================
def sca(objective_function, lb, ub, dim, num_agents, max_iter, seed=None):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(lb, ub, size=(num_agents, dim))
    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    convergence = np.zeros(max_iter)
    pop_history_2d = []

    a = 2.0
    for t in range(max_iter):
        # Evaluate
        positions = np.clip(positions, lb, ub)
        fitnesses = np.apply_along_axis(objective_function, 1, positions)

        # Update best
        idx = np.argmin(fitnesses)
        if fitnesses[idx] < dest_fitness:
            dest_fitness = float(fitnesses[idx])
            dest_pos = positions[idx].copy()

        convergence[t] = dest_fitness

        # Save for 2D slides (first 20 iterations)
        if dim >= 2 and t < 20:
            pop_history_2d.append(positions[:, :2].copy())

        # Move agents
        r1 = a - t * (a / max_iter)
        for i in range(num_agents):
            for j in range(dim):
                r2 = rng.uniform(0, 2*np.pi)
                r3 = rng.uniform(0, 2)
                r4 = rng.uniform(0, 1)
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + r1*np.sin(r2)*abs(r3*dest_pos[j] - positions[i, j])
                else:
                    positions[i, j] = positions[i, j] + r1*np.cos(r2)*abs(r3*dest_pos[j] - positions[i, j])

    logs = {"convergence": convergence, "pop_history_2d": pop_history_2d}
    return dest_pos, dest_fitness, logs


# =========================
# Plot helpers
# =========================
def plot_rosenbrock_contour(x_range=(-2, 2), y_range=(-2, 2), levels=50, out="contour_f2_rosenbrock.png"):
    xs = np.linspace(x_range[0], x_range[1], 400)
    ys = np.linspace(y_range[0], y_range[1], 400)
    XX, YY = np.meshgrid(xs, ys)
    Z = 100.0*(YY - XX**2)**2 + (1.0 - XX)**2

    plt.figure(figsize=(7, 6))
    cs = plt.contour(XX, YY, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.scatter([1],[1], marker="*", s=120, label="global min (1,1)")
    plt.title("Rosenbrock f2 (D=2) Contour")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def save_agent_slides(pop_history_2d, x_range=(-2, 2), y_range=(-2, 2), outdir="rosenbrock_agents"):
    os.makedirs(outdir, exist_ok=True)
    xs = np.linspace(x_range[0], x_range[1], 400)
    ys = np.linspace(y_range[0], y_range[1], 400)
    XX, YY = np.meshgrid(xs, ys)
    Z = 100.0*(YY - XX**2)**2 + (1.0 - XX)**2

    for it, pop in enumerate(pop_history_2d, start=1):
        plt.figure(figsize=(7, 6))
        cs = plt.contour(XX, YY, Z, levels=50)
        plt.clabel(cs, inline=True, fontsize=8)
        plt.scatter(pop[:, 0], pop[:, 1], s=25)  # default color
        plt.scatter([1],[1], marker="*", s=120, label="global min (1,1)")
        plt.title(f"Rosenbrock agents at iteration {it}")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
        plt.tight_layout()
        fname = os.path.join(outdir, f"iter_{it:02d}.png")
        plt.savefig(fname, dpi=200); plt.close()


def plot_convergence(curve, start_iter=5, end_iter=100, out="rosenbrock_convergence.png"):
    n = len(curve)
    s = max(0, min(start_iter-1, n-1))
    e = max(1, min(end_iter, n))
    xs = np.arange(s+1, e+1)
    plt.figure(figsize=(7, 5))
    plt.plot(xs, curve[s:e])
    plt.xlabel("Iteration")
    plt.ylabel("Best objective value")
    plt.title("Rosenbrock: Convergence (best-so-far)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def build_gif_from_png(folder="rosenbrock_agents", out_gif="rosenbrock_agents_slideshow.gif", duration=250):
    frames = []
    for i in range(1, 21):  # 20 frames
        fname = os.path.join(folder, f"iter_{i:02d}.png")
        frames.append(Image.open(fname))
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=duration, loop=0)


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Settings for Rosenbrock visuals
    D = 2
    lb, ub = -2.0, 2.0
    num_agents = 20
    max_iter = 120  # ensure >= 100 for convergence slice
    seed = 2025

    # 1) Contour
    plot_rosenbrock_contour(x_range=(lb, ub), y_range=(lb, ub))

    # 2) Run SCA and collect history
    _, best, logs = sca(rosenbrock_f2, lb, ub, D, num_agents, max_iter, seed=seed)
    print(f"[Rosenbrock D=2] best min = {best:.6e}")

    # 3) Slides for first 20 iterations
    save_agent_slides(logs["pop_history_2d"], x_range=(lb, ub), y_range=(lb, ub), outdir="rosenbrock_agents")
    try:
        build_gif_from_png(folder="rosenbrock_agents", out_gif="rosenbrock_agents_slideshow.gif", duration=250)
    except Exception:
        pass

    # 4) Convergence k=5..100
    plot_convergence(logs["convergence"], start_iter=5, end_iter=100, out="rosenbrock_convergence.png")
    print("Done.")
