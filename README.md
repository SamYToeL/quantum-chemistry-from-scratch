# Quantum Chemistry from Scratch

Implementing quantum chemistry methods from scratch in Python to explore electronic structure theory and numerical algorithms.

## Overview

This repository collects programming exercises, personal implementations, and reference materials for learning quantum chemistry. Python scripts and Jupyter notebooks connect the underlying equations with explicit numerical calculations. The emphasis is on understanding the algorithms and their implementation.

Here, "from scratch" refers to implementing the numerical steps of the methods. Molecular integrals are supplied by external quantum chemistry libraries where needed.

## Repository contents

| Directory | Topic | Contents |
| --- | --- | --- |
| [Common](Common/) | Shared resources | Common files used by the programming projects |
| [Project_01](Project_01/) | Molecular geometry | Molecular coordinates, bond lengths and angles, dihedral angles, center of mass, and moments of inertia |
| [Project_03](Project_03/) | Hartree–Fock | Closed-shell SCF, Fock matrix construction, Roothaan equations, density matrices, total energies, and dipole moments |
| [Project_04](Project_04/) | MP2 | Molecular-orbital integral transformation and second-order correlation energy |
| [Project_05](Project_05/) | CCSD learning materials | Tutorial notebook, reference solution, and example input data |

The repository includes both exercise notebooks and reference solutions. Some exercise cells intentionally contain `NotImplementedError`; use the corresponding implementation or solution files when looking for executable examples.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/SamYToeL/quantum-chemistry-from-scratch.git
cd quantum-chemistry-from-scratch
```

### 2. Prepare a Python environment

The numerical examples use NumPy and SciPy, and the notebooks require Jupyter. PySCF is used by some reference code; `Project_03_HF.py` uses `forte2` for molecular integrals. Install the dependencies required by the specific file you want to run.

For the numerical libraries, notebooks, and PySCF-based examples:

```bash
python -m pip install numpy scipy jupyterlab pyscf
```

The `forte2`-based examples additionally require a working `forte2` installation.

### 3. Explore the projects

```bash
jupyter lab
```

Start with `Project_01`, then move on to Hartree–Fock and MP2. Read the notebook instructions and imports before running the cells.

Several scripts and notebooks use relative input paths. Run them from the relevant project directory and check any working-directory setup cells or `os.chdir(...)` calls against your local checkout. For example, the HF script can be launched from its own directory:

```bash
cd Project_03
python Project_03_HF.py
```

## Learning goals

- Translate quantum chemistry equations into readable Python code.
- Understand matrix operations, tensor contractions, and orbital transformations.
- Follow the construction and convergence of SCF calculations.
- Explore how electron correlation is introduced beyond Hartree–Fock.

## Acknowledgments

The included exercises and reference materials draw on quantum chemistry programming tutorials, including the [Crawford Group Programming Projects](https://github.com/CrawfordGroup/ProgrammingProjects). Consult the individual notebooks and source files for their original references and credits. Personal implementations and supplied reference solutions are included together for learning and comparison.
