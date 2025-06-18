# HPC Project: Epidemic Simulation

## Introduction

This repository provides a high-performance implementation of the epidemic propagation model using the Susceptible-Infected-Recovered (SIR) framework, as specified in the assignment proposal (docs dir). The simulation lets you model the spread of an infectious disease across a moving population in a 2D region, with parameters for infection radius, contagion factor, recovery probability, incubation period, and more.

## Features

- **Serial** (single-threaded) implementation  
- **OpenMP**-parallelized implementation   
- **CUDA**-accelerated implementation  
- Scripts for automated testing and visualization

## What to do before running

Before you build or launch any simulations, you need to create the following directories inside the `src/` folder. These will organize your binaries, reports and test data:

```bash
cd src
mkdir bin report test_cuda test_serial test_omp
```

Slurm scripts are provided in the `src/` directory.

Happy Simulating! 😀