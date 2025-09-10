# About The Project
Convergent-divergent nozzles are critical in generating useful thrust by accelerating combustion chamber gasses to supersonic speeds. This user friendly simulator allows a designer to analyze flow properties all along the nozzle as well as analyze tradeoffs due to design changes. The simulator takes into account back pressure to calculate normal shock locations in the divergent section and determine if flow is choked or subsonic.

## Theory
Initially the code assumes 1D steady adiabatic isentropic flow with an ideal gas of constant specific heats. Supersonic and subsonic flow is calculated and compared with back pressure P_e to determine if pressure matching is satisfied. If pressure lies between these two ranges, the code sweeps through the divergent section plotting normal shocks and calculating post-shockwave properties. Exit pressure is calculated using the area-mach relation and isentropic pressure relation. In the case that exit pressure is lower than back pressure even with a shock at exit, an overexpanded exhaust with oblique shocks and Mach diamonds generated in the plume is assumed. If back pressure is lower than supersonic exit pressure, an underexpanded flow is assumed.

# Getting Started

## Installation
The code is built with Python 3.12 and requires the following modules listed in requirements.txt

1.  Clone the repository:
    ```bash
    git clone https://github.com/am975eng/Convergent-Divergent-Nozzle-Simulator.git
    cd Convergent-Divergent-Nozzle-Simulator
    ```

2.  (Recommended) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
