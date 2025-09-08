# SCA-Benchmark-Visualizations

This repository contains Python scripts that implement the Sine Cosine Algorithm (SCA) to optimize various benchmark mathematical functions. Additionally, it provides tools for visualizing the optimization process, including contour plots, agent movement, and convergence graphs.

## 🌟 Overview

The core of this project is the demonstration of the Sine Cosine Algorithm, a nature-inspired metaheuristic optimization algorithm. The effectiveness of SCA is tested on several standard benchmark functions, each with its own unique challenges for optimization algorithms. The scripts are organized to not only perform the optimization but also to generate a variety of plots that help in understanding the behavior of the algorithm.

## ✨ Features

  - **Sine Cosine Algorithm (SCA):** A core implementation of the SCA is provided.
  - **Benchmark Functions:** The algorithm is tested on the following standard benchmark functions:
      - **Sphere Function:** A simple, convex, and unimodal function.
      - **Rosenbrock Function:** A non-convex function with a narrow, parabolic-shaped global optimum valley.
      - **Rastrigin Function:** A highly multimodal function with a regular distribution of local optima.
      - **Griewank Function:** A multimodal function with a product term, making it difficult to optimize.
  - **Comprehensive Visualization:** The project generates several types of plots for in-depth analysis:
      - **Contour Plots:** To visualize the landscape of the benchmark functions in 2D.
      - **Agent Position Slideshows:** To track the movement of the search agents across the solution space over iterations.
      - **Convergence Curves:** To show the progression of the best-found solution over time.
      - **Statistical Analysis:** Plots showing the mean and standard deviation of results over multiple runs.

-----

## 🚀 Getting Started

To get started with this project, you'll need to have Python and the following libraries installed:

  - **NumPy:** For numerical operations.
  - **Matplotlib:** For generating the plots.
  - **Pillow:** For creating GIFs from the generated agent position images.

You can install these dependencies using pip:

```bash
pip install numpy matplotlib pillow
```

### 📂 File Descriptions

Here's a breakdown of the Python scripts included in this repository:

  - `sca_sphere_f1.py`: Implements SCA for the Sphere function. It runs multiple trials, performs statistical analysis, and generates visualizations for a 2D version of the function.
  - `sca_rosenbrock_f2.py`: Focuses on visualizing SCA's performance on the 2D Rosenbrock function, creating a contour plot, a GIF of agent movement, and a convergence graph.
  - `sca_rastrigin_f3.py`: Applies SCA to the Rastrigin function. This script is set up to run multiple trials and visualize the agent positions at a specific iteration.
  - `sca_griewank_f4.py`: Tackles the Griewank function. Similar to the Sphere example, it runs multiple trials for statistical analysis and also provides a 2D visualization demo.

-----

## 🏃‍♂️ How to Run the Scripts

You can run each script directly from your terminal. Here's how you can execute each one and what to expect as output:

### 🌐 Sphere Function (`sca_sphere_f1.py`)

This script will first run 10 trials of SCA on the 30-dimensional Sphere function and print the mean and standard deviation of the final results. It will then generate a set of visualizations for the 2D version of the function.

```bash
python sca_sphere_f1.py
```

**Generated Files:**

  - `f1_contour.png`: A contour plot of the 2D Sphere function.
  - `slides_f1/`: A directory containing images of the agent positions for the first 20 iterations.
  - `f1_convergence.png`: A graph showing the convergence of the algorithm.
  - `f1_mean_std.png`: A plot of the average minimum value with standard deviation over 10 runs.

### 🍌 Rosenbrock Function (`sca_rosenbrock_f2.py`)

This script is designed to provide a detailed look at SCA's behavior on the challenging Rosenbrock function.

```bash
python sca_rosenbrock_f2.py
```

**Generated Files:**

  - `contour_f2_rosenbrock.png`: A contour plot of the 2D Rosenbrock function.
  - `rosenbrock_agents/`: A directory with images of agent positions for the first 20 iterations.
  - `rosenbrock_agents_slideshow.gif`: An animated GIF showing the movement of the agents.
  - `rosenbrock_convergence.png`: A plot illustrating the convergence of the algorithm.

### 📈 Rastrigin Function (`sca_rastrigin_f3.py`)

This script will run SCA on the highly multimodal Rastrigin function. It is set up to perform 20 runs and will display a plot of the agent positions at a specified iteration for each run.

```bash
python sca_rastrigin_f3.py
```

**Output:**

  - The script will print the best fitness found in each of the 20 runs.
  - For each run, a plot will be displayed showing the agent positions on the Rastrigin contour map at the specified iteration.

### 🌍 Griewank Function (`sca_griewank_f4.py`)

Similar to the Sphere function script, this will perform a statistical analysis over 10 trials on the 30-dimensional Griewank function and then generate visualizations for the 2D case.

```bash
python sca_griewank_f4.py
```

**Generated Files:**

  - `f4_contour.png`: A contour plot of the 2D Griewank function.
  - `slides_f4/`: A directory containing images of agent movements.
  - `f4_convergence.png`: The convergence graph for the 2D demo.
  - `f4_mean_std.png`: The plot of the average minimum value with standard deviation over 10 runs.

-----

## 🤝 Contributing

Contributions to this project are welcome\! If you have suggestions for improvements, new features, or find any bugs, please feel free to open an issue or submit a pull request.

-----

This video provides a great overview of the LS engine, which is a popular choice for engine swaps and high-performance builds.
