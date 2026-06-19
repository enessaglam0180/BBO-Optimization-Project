# BBO: Beaver Behavior Optimizer for Solar PV Parameter Estimation and Benchmark Functions

This repository contains a high-performance Python framework implementing the **Beaver Behavior Optimizer (BBO)**. The framework is designed to evaluate standard continuous mathematical benchmark functions and solve real-world complex engineering optimization problems, specifically focusing on **Solar Photovoltaic (PV) Cell Parameter Estimation**.

---

## Theoretical Background

### 1. Beaver Behavior Optimizer (BBO)
BBO is a nature-inspired metaheuristic optimization algorithm based on the biological and ecological behaviors of beavers. It models their highly structured survival and environmental engineering strategies:
- **Dam Building & Habitat Modification:** Agents (beavers) structurally alter their positions to find optimal regions (secure ponds), representing the exploitation of high-quality solutions.
- **Foraging & Tree Felling:** Simulates the search for resource allocation paths, driving global exploration across the multi-dimensional search space.
- **Territory Defense & Lodging:** Maintains population diversity and prevents premature convergence by executing strategic relocations when environmental fitness declines.

### 2. Engineering Application: Solar PV Modeling
The practical module estimates the unknown electrical parameters of solar cells (such as the single-diode model) by minimizing the root-mean-square error (RMSE) between measured experimental currents and simulated theoretical currents using the optimized beaver search trajectories.

---

## Repository Structure

* **`src/BBO.py`**: Implementation of the core Beaver Behavior Optimizer mechanics, controlling foraging choices, dam site selections, and allocation movements.
* **`src/PSO.py`**: Standard Particle Swarm Optimization script utilized as a baseline competitor.
* **`src/GWO.py`**: Standard Grey Wolf Optimizer script utilized as a secondary baseline competitor.
* **`src/benchmark_functions.py`**: Contains standard continuous mathematical benchmarks (e.g., Sphere, Rosenbrock, Rastrigin, Griewank) used to test exploration and exploitation capabilities.
* **`src/compare_algorithms.py`**: Automated script to execute cross-algorithm performance evaluations.
* **`src/collect_statistics.py`**: Aggregates statistical metrics across multiple independent runs (Mean, Best, Worst, Standard Deviation) to ensure scientific validity.
* **`src/plot_results.py`**: Visualizes convergence curves, error graphs, and performance metrics.
* **`src/main.py`**: The central execution script containing optimization pipelines for both benchmarks and solar parameter estimation.
* **`final_results.csv`**: Contains exported data summaries of the statistical evaluations.
* **`Comparison_Result.png` & `Solar_PV_Results.png`**: Analytical plots displaying algorithm convergence trajectories and tracking precision.

---

## Installation & Setup

Ensure you have a Python 3.8+ environment installed. You can set up the required data science and visualization dependencies with the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

## How to Run

### 1. Run the Complete Framework

To run the default pipeline which tests the BBO algorithm on both the benchmark optimization tasks and the solar PV cell calculation models, run:

```bash
python src/main.py

```

### 2. Run Comparative Studies

To trigger an execution loop that directly compares the efficiency, computational speed, and stability of BBO against PSO and GWO, run:

```bash
python src/compare_algorithms.py

```

---

## Hyperparameter Configurations

You can customize the algorithmic hyperparameters within `src/main.py` or individual source files to experiment with performance:

| Parameter | Description | Typical Settings |
| --- | --- | --- |
| `Population Size` | Total number of beavers (candidate solutions) within the colony | $20 - 50$ |
| `Max Iterations` | Maximum generation cycles for environmental optimization | $100 - 500$ |
| `Exploration Factor` | Controls the balance between foraging wide areas and dam stabilization | $0.2 - 0.5$ |

---

## Results & Visualizations

The execution outputs logs to the console and exports high-quality figures:

1. **Convergence Curves (`Comparison_Result.png`):** Shows the minimization trajectory of fitness values across iterations, mapping out how BBO scales against PSO and GWO.
2. **Solar PV Calibration Charts (`Solar_PV_Results.png`):** Displays the simulated I-V or P-V curves using the optimized parameters overlaid with baseline metrics to show fitting accuracy.
3. **Statistical Integrity (`final_results.csv`):** Retains tabular metrics proving the stability, standard deviation, and best-found solutions across independent test iterations.

---

## License

This repository is open-source. Feel free to clone, modify, and utilize it for academic research or industrial optimization projects.
