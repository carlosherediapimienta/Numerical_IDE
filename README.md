# Integral-Differential Equation Solver

## Description

This project provides numerical solutions for first and second-order integral-differential equations using the `scipy.integrate.solve_ivp` for differential equation solving and `scipy.integrate.quad` for numerical integration. Designed to be easily extendable for more complex cases and different types of integral-differential equations, it aims to offer a flexible tool for academic and research purposes.

## Features

- Solves first and second-order integral-differential equations numerically.
- Utilizes `solve_ivp` from SciPy for robust differential equation solving.
- Employs `quad` for accurate numerical integration over specified intervals.
- Plots solutions using Matplotlib to visualize equation behavior over time.

## Getting Started

### Prerequisites

- Python 3.6 or later
- SciPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install scipy matplotlib
```

### Installation

Clone this repository or download the source code to your local machine. No further installation is required as the project uses standard Python libraries.

### Usage

To use the equation solvers, import the desired class (`ide_solver_1rst_order` for first-order equations or `ide_solver_2nd_order` for second-order equations) into your Python script and instantiate it with the appropriate parameters:

```python
from your_module_name import ide_solver_1rst_order, ide_solver_2nd_order

# Define your differential and integral functions, initial conditions, and time span here
# Example for a first-order equation
solver = ide_solver_1rst_order(f, g, y0, t_span, a, b)
solution = solver.solve()
solver.plot_solution()

# Example for a second-order equation
solver2 = ide_solver_2nd_order(f, g, y0, z0, t_span, a, b)
solution2 = solver2.solve()
solver2.plot_solution()
```

Replace `your_module_name` with the actual name of the Python file containing the solver classes.

## Contributing

Contributions to enhance the functionality or efficiency of the solvers are welcome. Please feel free to fork the repository and submit pull requests.

## Versioning

We intend to update and improve the solver classes regularly. Check the repository for the latest version.

## Authors

- Your Name or Your Organization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The SciPy community for providing the powerful numerical integration and solving tools used in this project.
- Anyone whose code was used as inspiration or directly incorporated into this project.


