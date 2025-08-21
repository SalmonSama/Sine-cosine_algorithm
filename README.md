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
````

### 2\. Multimodal Benchmark Functions

Multimodal functions have multiple local optima, which makes them suitable for testing an algorithm's ability to avoid getting trapped in a local minimum and to explore the search space effectively.

#### Shifted Schwefel Function (F8)

The Schwefel function is a complex multimodal function with many local minima.

**Formula:**
$ f(x) = \\sum\_{i=1}^{D} -x\_i \\sin(\\sqrt{|x\_i|}) $

**Implementation (`Multimodal_benchmark_functions.py`):**

```python
def shifted_schwefel_f8(x):
    """
    Schwefel function F8 with a shifted minimum.
    f(x) = sum(-x_i * sin(sqrt(|x_i|)))
    """
    # Create the shift vector o = [-300, -300, ..., -300]
    shift_vector = np.full_like(x, -300.0)

    shifted_x = x - shift_vector

    # Calculate the sum for the shifted vector based on the formula
    return np.sum(-shifted_x * np.sin(np.sqrt(np.abs(shifted_x))))
```

### 3\. Composite Benchmark Functions

Composite functions are designed to be challenging by having a complex structure with multiple local optima.

#### Shifted Rastrigin Function (F9)

The Rastrigin function is a classic example of a multimodal function with a regular distribution of local minima.

**Formula:**
$ f(x) = \\sum\_{i=1}^{D} (x\_i^2 - 10 \\cos(2 \\pi x\_i) + 10) $

**Implementation (`Composite_benchmark_functions.py`):**

```python
def shifted_rastrigin_f9(x):
    """
    Rastrigin function F9 with a shifted minimum.
    f(x) = sum(x_i^2 - 10 * cos(2 * pi * x_i) + 10)
    """
    # Create the shift vector o = [-2, -2, ..., -2]
    shift_vector = np.full_like(x, -2.0)

    shifted_x = x - shift_vector

    # Calculate the sum for the shifted vector based on the formula
    return np.sum(np.square(shifted_x) - 10 * np.cos(2 * np.pi * shifted_x) + 10)
```

## How to Run

Each benchmark test is contained in its own Python script. To run a specific test, navigate to the repository's directory in your terminal and execute the desired script.

**1. Run the Unimodal Benchmark Test:**

```bash
python Unimodal_benchmark_functions.py
```

**2. Run the Multimodal Benchmark Test:**

```bash
python Multimodal_benchmark_functions.py
```

**3. Run the Composite Benchmark Test:**

```bash
python Composite_benchmark_functions.py
```

## Results

After running each script, the output will display the progress of the optimization process at every 100 iterations. The final results will show:

  - The best solution (vector) found by the SCA.
  - The best fitness value corresponding to that solution.
  - The theoretical minimum value ($f\_{min}$) for the benchmark function.
  - The difference (error) between the found fitness and the theoretical minimum.

The goal is to minimize this error, indicating that the algorithm has successfully located the global optimum.

## Dependencies

  - **Python 3.x**
  - **NumPy:** A fundamental package for scientific computing in Python.
  - **math:** A standard Python library for mathematical functions.

You can install NumPy using pip:

```bash
pip install numpy
```

