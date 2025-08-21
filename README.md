# Sine Cosine Algorithm for Global Optimization Benchmarking

![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository provides a Python implementation of the Sine Cosine Algorithm (SCA), a metaheuristic optimization algorithm, tested on a set of unimodal, multimodal, and composite benchmark functions. The purpose is to demonstrate the effectiveness of the SCA in navigating different types of optimization landscapes to find the global optimum.

## About the Sine Cosine Algorithm (SCA)

The Sine Cosine Algorithm is a population-based optimization algorithm that utilizes sine and cosine functions to explore and exploit the search space. The algorithm creates a set of random initial solutions and then encourages them to move towards the best solution found so far. The position-updating equations are as follows:

$ X_{i}^{t+1} = X_{i}^{t} + r_1 \times \sin(r_2) \times |r_3 P_{i}^{t} - X_{i}^{t}| $

$ X_{i}^{t+1} = X_{i}^{t} + r_1 \times \cos(r_2) \times |r_3 P_{i}^{t} - X_{i}^{t}| $

These two equations are combined as follows:

$$
X_{i}^{t+1} = \begin{cases} X_{i}^{t} + r_1 \times \sin(r_2) \times |r_3 P_{i}^{t} - X_{i}^{t}|, & r_4 < 0.5 \\ X_{i}^{t} + r_1 \times \cos(r_2) \times |r_3 P_{i}^{t} - X_{i}^{t}|, & r_4 \geq 0.5 \end{cases}
$$

Where:
- $X_{i}^{t}$ is the position of the current solution in the i-th dimension at the t-th iteration.
- $P_{i}^{t}$ is the position of the destination point in the i-th dimension.
- $r_1, r_2, r_3, r_4$ are random numbers that guide the optimization process.

## Benchmark Functions

The SCA is tested on three different categories of benchmark functions to evaluate its performance in various optimization scenarios.

### 1. Unimodal Benchmark Functions

Unimodal functions have a single global optimum, making them ideal for testing the exploitation capability of an algorithm.

#### Shifted Sphere Function (f1)

The Sphere function is a simple, convex, and unimodal function. A shift is introduced to displace the minimum from the origin.

**Formula:**
$ f(x) = \sum_{i=1}^{D} (x_i - o_i)^2 $

Where 'o' is the shifted position.

**Implementation (`Unimodal_benchmark_functions.py`):**
```python
def shifted_sphere_f1(x):
    """
    Sphere function f1 with a shifted minimum based on Table A.1.
    f(x) = sum((x_i - o_i)^2)
    """
    # Create the shift vector o = [-30, -30, ..., -30]
    shift_vector = np.full_like(x, -30.0)

    # Calculate the sum of squares for the shifted vector
    return np.sum(np.square(x - shift_vector))
