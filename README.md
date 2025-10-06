# About The Project
**Convergent-divergent nozzle simulator** is a Python GUI that generates thermodynamic properties for a variety of nozzle geometries as well as optimize design for an ideal thrust.

# Table of Contents
- [Theory](#theory)
- [Getting Started](#getting-started)
- [Installation](#installation)

## Theory
Initially the code assumes **1D steady adiabatic isentropic** flow with an ideal gas of constant specific heats. The code begins by calculating supersonic and subsonic critical exit pressures by using the area-Mach relation assuming choked flow. It then compares ambient pressure to critical pressure conditions to determine the type of flow produced. 
If ambient pressure is above chamber pressure then reverse flow is produced.
If ambient pressure is above the subsonic limit, the code assumes subsonic flow throughout the entire nozzle.
If ambient pressure is less than subsonic, the code sweeps through the divergent section plotting normal shocks and calculating post-shockwave properties at each location trying to match ambient conditions.
If a nozzle at exit can not bring pressure low enough, then the nozzle is overexpanded and Prandtl-Meyer shockwaves are plotted in the exhaust plume. 
If ambient pressure is lower than supersonic exit pressure then the nozzle is underexpanded with oblique shockwaves produced in the exhaust.

Once flow type is established, the code uses isentropic relations, stagnation conditions, and area-Mach relation to generate Mach, temperature, and pressure curves at each contour point.

Method of characteristics is a type of solution method for solving partial differential equations through reduction to an ordinary differential equation. We initially begin with the continuity equation and Euler's equation derived from the inviscid assumption for Navier-Stokes equations. The velocity potential is derived from these equations and is of the form (1). At every point A, the slope of the characteristic line is given as (2) in which the PDE reduces to compatibility equations.

# Getting Started

## Installation
The code is built with Python 3 and dependencies can be installed using the following.

1.  Clone the repository:
    ```bash
    git clone https://github.com/am975eng/Convergent-Divergent-Nozzle-Simulator.git
    cd Convergent-Divergent-Nozzle-Simulator
    ```

2.  (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv test_venv
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass # If first time using venv
    test_venv\Scripts\activate.ps1  # Windows
    ```

3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Run main script
    ```bash
    python main.py
    ```