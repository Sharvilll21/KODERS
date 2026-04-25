# NumCompute – A Modular Scientific Computing Toolkit

NumCompute is a small, from scratch scientific computing and machine learning toolkit built using only Python and NumPy.  
The aim of the project is not just to “use” ML concepts, but to understand and implement them at a lower level—how data flows, how computations are structured, and how performance and numerical stability are handled in practice.

Instead of relying on libraries like scikit-learn, this project rebuilds key ideas in a simplified, modular way so each component remains transparent and easy to reason about.

---

## What this project focuses on

A few principles guided the design:

- **Vectorised computation** wherever possible, avoiding Python loops in core logic  
- **Numerical stability**, including safe handling of edge cases like NaNs, zero variance, and large values  
- **Clear, consistent APIs**, so different components can plug into each other without friction  
- **Modularity**, allowing individual parts (e.g., preprocessing, metrics, pipeline) to be developed and tested independently  

The result is a toolkit that is simple in scope, but deliberate in design.

---

## Current capabilities

The project currently supports:

- A lightweight **pipeline abstraction** for chaining transformations and models  
- Basic **preprocessing tools** (e.g., standard scaling)  
- A simple **linear regression model** implemented using NumPy  
- **Evaluation metrics**, including mean squared error  
- A small **benchmarking utility** to compare vectorised operations against Python loops  

These components are integrated into a working end-to-end workflow demonstrated in the notebook.

---

## Project structure
