# -*- coding: utf-8 -*-
"""
SCA on Rosenbrock (f2) — D=2 visuals & Enhanced Analysis (Revised)
-------------------------------------------------------------------
Generates:
1) Agent slideshow (first 20 iterations, 20 agents) -> rosenbrock_agents/iter_XX.png
2) A directory 'rosenbrock_analysis_plots' containing:
   - Contour plot (D=2) -> contour_f2_rosenbrock.png
   - Search history of all agents (first 100 iters) -> 1_search_history.png
   - Trajectory of the first variable of the first agent (first 100 iters) -> 2_first_agent_trajectory.png
   - Average fitness of the population (first 100 iters) -> 3_average_fitness.png
   - Full Convergence curve (first 100 iters) -> 4_convergence_curve.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.collections import LineCollection

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
# Sine Cosine Algorithm (Modified to log more data)
# =========================
def sca(objective_function, lb, ub, dim, num_agents, max_iter, seed=None):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(lb, ub, size=(num_agents, dim))
    dest_pos = np.zeros(dim)
    dest_fitness = float("inf")

    # --- Logging variables ---
    convergence = np.zeros(max_iter)
    pop_history_2d = []
    avg_fitness_history = np.zeros(max_iter)
    first_agent_traj_x1 = np.zeros(max_iter)
    full_pop_history = []

    a = 2.0
    for t in range(max_iter):
        # Evaluate
        positions = np.clip(positions, lb, ub)
        fitnesses = np.apply_along_axis(objective_function, 1, positions)

        # --- Log data for this iteration ---
        avg_fitness_history[t] = np.mean(fitnesses)
        first_agent_traj_x1[t] = positions[0, 0]
        full_pop_history.append(positions.copy())

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
                r2 = rng.uniform(0, 2 * np.pi)
                r3 = rng.uniform(0, 2)
                r4 = rng.uniform(0, 1)
                if r4 < 0.5:
                    positions[i, j] = positions[i, j] + r1 * np.sin(r2) * abs(r3 * dest_pos[j] - positions[i, j])
                else:
                    positions[i, j] = positions[i, j] + r1 * np.cos(r2) * abs(r3 * dest_pos[j] - positions[i, j])

    logs = {
        "convergence": convergence,
        "pop_history_2d": pop_history_2d,
        "avg_fitness": avg_fitness_history,
        "first_agent_traj_x1": first_agent_traj_x1,
        "full_pop_history": full_pop_history
    }
    return dest_pos, dest_fitness, logs


# =======================================
# New Plotting Functions (Iterations limited to 100)
# =======================================

def plot_search_history(history, x_range=(-2, 2), y_range=(-2, 2), out="search_history.png"):
    """ 1. Plots the trajectory of all search agents on the contour plot (first 100 iters). """
    xs = np.linspace(x_range[0], x_range[1], 400)
    ys = np.linspace(y_range[0], y_range[1], 400)
    XX, YY = np.meshgrid(xs, ys)
    Z = 100.0 * (YY - XX**2)**2 + (1.0 - XX)**2

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.contour(XX, YY, Z, levels=50, cmap='viridis', alpha=0.7)
    
    # Slice history to the first 100 iterations
    history_sliced = history[:100]
    agent_trajectories = np.array(history_sliced).transpose(1, 0, 2)
    
    for agent_path in agent_trajectories:
        points = agent_path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, color='red', alpha=0.3, linewidth=1)
        ax.add_collection(lc)

    ax.scatter([1], [1], marker="*", s=150, c='gold', edgecolors='black', zorder=5, label="Global Min (1,1)")
    ax.set_title("Search History of All Agents (First 100 Iterations)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend()
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()

def plot_first_agent_trajectory(trajectory, out="first_agent_trajectory.png"):
    """ 2. Plots the trajectory of x1 for the first agent (first 100 iters). """
    plt.figure(figsize=(8, 5))
    # Slice trajectory to the first 100 iterations
    plt.plot(trajectory[:100])
    plt.xlabel("Iteration")
    plt.ylabel("Value of x1 for the first agent")
    plt.title("Trajectory of the First Variable (x1) of the First Agent (First 100 Iterations)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def plot_average_fitness(avg_fitness, out="average_fitness.png"):
    """ 3. Plots the average fitness of all search agents (first 100 iters). """
    plt.figure(figsize=(8, 5))
    # Slice fitness data to the first 100 iterations
    plt.plot(avg_fitness[:100])
    plt.xlabel("Iteration")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness of Search Agents (First 100 Iterations)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def plot_full_convergence_curve(curve, out="convergence_curve.png"):
    """ 4. Plots the convergence curve (first 100 iters). """
    plt.figure(figsize=(8, 5))
    # Slice convergence data to the first 100 iterations
    plt.plot(curve[:100])
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Value (Fitness)")
    plt.title("Convergence Curve (First 100 Iterations)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

# =========================
# Original Plot helpers
# =========================
def plot_rosenbrock_contour(x_range=(-2, 2), y_range=(-2, 2), levels=50, out="contour_f2_rosenbrock.png"):
    xs = np.linspace(x_range[0], x_range[1], 400)
    ys = np.linspace(y_range[0], y_range[1], 400)
    XX, YY = np.meshgrid(xs, ys)
    Z = 100.0 * (YY - XX**2)**2 + (1.0 - XX)**2

    plt.figure(figsize=(7, 6))
    cs = plt.contour(XX, YY, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.scatter([1], [1], marker="*", s=120, label="global min (1,1)")
    plt.title("Rosenbrock f2 (D=2) Contour")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()

def save_agent_slides(pop_history_2d, x_range=(-2, 2), y_range=(-2, 2), outdir="rosenbrock_agents"):
    os.makedirs(outdir, exist_ok=True)
    xs = np.linspace(x_range[0], x_range[1], 400)
    ys = np.linspace(y_range[0], y_range[1], 400)
    XX, YY = np.meshgrid(xs, ys)
    Z = 100.0 * (YY - XX**2)**2 + (1.0 - XX)**2

    for it, pop in enumerate(pop_history_2d, start=1):
        plt.figure(figsize=(7, 6))
        cs = plt.contour(XX, YY, Z, levels=50)
        plt.clabel(cs, inline=True, fontsize=8)
        plt.scatter(pop[:, 0], pop[:, 1], s=25)
        plt.scatter([1], [1], marker="*", s=120, label="global min (1,1)")
        plt.title(f"Rosenbrock agents at iteration {it}")
        plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
        plt.tight_layout()
        fname = os.path.join(outdir, f"iter_{it:02d}.png")
        plt.savefig(fname, dpi=200); plt.close()

# These functions are no longer called in the main block but are kept for reference
def plot_convergence(curve, start_iter=5, end_iter=100, out="rosenbrock_convergence.png"):
    n = len(curve)
    s = max(0, min(start_iter - 1, n - 1))
    e = max(1, min(end_iter, n))
    xs = np.arange(s + 1, e + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(xs, curve[s:e])
    plt.xlabel("Iteration")
    plt.ylabel("Best objective value")
    plt.title("Rosenbrock: Convergence (best-so-far from iter 5-100)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def build_gif_from_png(folder="rosenbrock_agents", out_gif="rosenbrock_agents_slideshow.gif", duration=250):
    frames = []
    for i in range(1, 21):
        fname = os.path.join(folder, f"iter_{i:02d}.png")
        if os.path.exists(fname):
            frames.append(Image.open(fname))
    if frames:
        frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=duration, loop=0)

# =========================
# Main Execution (Revised)
# =========================
if __name__ == "__main__":
    # Settings
    D = 2
    lb, ub = -2.0, 2.0
    num_agents = 20
    max_iter = 120 # Still run 120 iterations to get complete data
    seed = 2025

    # --- Create output directories ---
    output_dir = "rosenbrock_analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Run SCA once and collect all logs ---
    _, best_fitness, logs = sca(rosenbrock_f2, lb, ub, D, num_agents, max_iter, seed=seed)
    print(f"[Rosenbrock D=2] Final best fitness = {best_fitness:.6e}")

    # --- Generate agent slide images ---
    # These are PNGs showing agent positions for the first 20 iterations
    save_agent_slides(logs["pop_history_2d"], x_range=(lb, ub), y_range=(lb, ub), outdir="rosenbrock_agents")
    print("\nSuccessfully generated agent slides in 'rosenbrock_agents/' directory.")
    
    # --- Generate all analysis plots in the 'rosenbrock_analysis_plots' directory ---
    
    # 1. Contour plot (Moved to analysis folder)
    plot_rosenbrock_contour(
        x_range=(lb, ub), 
        y_range=(lb, ub),
        out=os.path.join(output_dir, "contour_f2_rosenbrock.png")
    )
    
    # 2. Search history
    plot_search_history(
        logs["full_pop_history"],
        x_range=(lb, ub),
        y_range=(lb, ub),
        out=os.path.join(output_dir, "1_search_history.png")
    )
    
    # 3. Trajectory of the first agent's first variable
    plot_first_agent_trajectory(
        logs["first_agent_traj_x1"],
        out=os.path.join(output_dir, "2_first_agent_trajectory.png")
    )
    
    # 4. Average fitness
    plot_average_fitness(
        logs["avg_fitness"],
        out=os.path.join(output_dir, "3_average_fitness.png")
    )
    
    # 5. Full convergence curve
    plot_full_convergence_curve(
        logs["convergence"],
        out=os.path.join(output_dir, "4_convergence_curve.png")
    )

    print(f"\nSuccessfully generated analysis plots in the '{output_dir}' directory.")