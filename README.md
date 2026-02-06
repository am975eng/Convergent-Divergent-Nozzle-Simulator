# About The Project
**Convergent-divergent Nozzle Simulator** is a Python GUI that computes thermodynamic properties for various nozzle geometries and optimizes the nozzle design to achieve ideal thrust.
<p align="center">
    <img src="./assets/CD_Nozzle_Intro.gif" width="720">
</p>        
<p align="center">
    Figure 1: Nozzle GUI
</p>
<p align="center">
    <img src="./assets/MOC_Mesh.PNG" width="720">
</p>
<p align="center">
    Figure 2: Mesh of flow points using method of characteristics method.    
</p>

## Features
- 1D, steady, adiabatic, isentropic flow solver
- Thrust design using gradient descent algorithm with ADAM optimizer 
- Prandtl-Meyer and oblique shockwave plotting for overexpanded and underexpanded jets respectively
- Method of characteristics nozzle contour generator
- Cross-platform support (Windows/macOS/Linux) PyQt6 GUI

# Table of Contents
- [Theory](#theory)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Basic Usage](#basic-usage)

## Theory
### Flow Regimes
Initially the code assumes **1D steady adiabatic isentropic** flow with an ideal gas of constant specific heats. The code generates either a method of characteristics, conical, or Rao bell nozzle based on user input. The code then calculates supersonic and subsonic critical exit pressures by using the area-Mach relation assuming choked flow. Comparing ambient pressure to critical pressure conditions determines flow type.

If ambient pressure is above chamber pressure, then reverse flow is produced.

If ambient pressure is above the subsonic limit, the code assumes subsonic flow throughout the entire nozzle.

If ambient pressure is less than subsonic, the code sweeps through the divergent section plotting normal shocks and calculating post-shockwave properties at each location trying to match ambient conditions.

If a nozzle at exit can not bring pressure low enough, then the nozzle is overexpanded and Prandtl-Meyer shockwaves are plotted in the exhaust plume. 

If ambient pressure is lower than supersonic exit pressure then the nozzle is underexpanded with oblique shockwaves produced in the exhaust.

Once flow type is established, the code uses isentropic relations, stagnation conditions, and area-Mach relation to generate Mach, temperature, and pressure curves at each contour point.

<p align="center">
    <img src="./assets/Nozzle_Drawing.png" width=50%>
</p>
<p align="center">
    Figure 3: Pressure variation from chamber to exhaust with critical pressure conditions under various flow regimes.
</p>

### Method of Characteristics
Method of characteristics is a solution method for solving partial differential equations through reduction to an ordinary differential equation. We initially begin with the continuity equation and Euler's equation derived from the inviscid assumption for Navier-Stokes equations. The velocity potential is derived from these equations and is of the form

$$\left(1 - \frac{\Phi_x^2}{a^2}\right) \Phi_{xx} + \left(1 - \frac{\Phi_y^2}{a^2}\right) \Phi_{yy}-2\,\frac{\Phi_x \Phi_y}{a^2}\, \Phi_{xy}= 0 $$

For a hyperbolic PDE, at every point A, there exists characteristic lines in which the PDE reduce to ODE compatiblity equations. The direction of these characteristic or Mach lines is given as,

$$\left(\frac{dy}{dx}_{char} = \tan{\theta \mp \mu} \right)$$

The compatibility equation for these char lines with K- and K+ representing right and left running characateristic lines respectfully are,

$$\theta + \nu(M) = const = K-$$
$$\theta - \nu(M) = const = K+$$

Starting with initially defined fluid properties, we can generate characteristic lines and solve for fluid properties at intersections by setting characteristic strengths equal to each other. After calculating theta and Prandtl-Meyer angle we can find local Mach number and flow deflection angle. Using the slopes and location of left and right running char lines, we can find the current point's location. For centerline points, we use the condition that flow must flow at zero angle. Along the wall we find the intersection between an average of current and previous flow angles with the left running char line. This ensures shockwave is absorbed by the contour and does not propagate further. We continue downmarching through intersection points until we reach the exit.

<p align="center">
    <img src="./assets/Char_line.PNG" width=50%>
</p>
<p align="center">
    Figure 4: Characteristic lines and streamline at point A. Source: ANSYS, Inc. (2020). "Lesson 6: Nozzle Tutorial Handout." ANSYS Innovation Courses. Retrieved from [https://innovationspace.ansys.com/courses/wp-content/uploads/2020/12/Lesson6-Handout-NT-v1.pdf]
</p>

### Optimization/Depress
<p align="center">
    <img src="./assets/CD_Nozzle_Mid.gif" width="720">
</p>   
The optimizer allows a designer to generate a thruster geometry given an ideal design thrust. The function uses a gradient descent algorithm initially calculating the gradient of the cost function with regards to throat radius. An adaptive learning rate is calculated factoring in the gradient and current cost to ensure subsequent step sizes are proportional. The ADAM optimizer then produces an updated throat radius which continues to iterate until the ideal thrust is reached.

Depressurization uses Dean Wheeler's blowdown model for analyzing tank properties during choked flow. A transient mass balance yields the following equation when factoring in ideal gas law and mass flow rate,

$$V_{\text{tank}} \frac{d \rho_{\text{tank}}}{dt} 
= -C_d A_* \, \rho_{\text{tank}} 
\left( \gamma \frac{R T_{\text{tank}}}{M} \right)^{\frac{1}{2}} 
\left( \frac{2}{\gamma + 1} \right)^{\frac{\gamma + 1}{2(\gamma - 1)}} $$

All the constants are readily available with the exception of tank temperature which can be calculated under two assumptions.

Isothermal assumption assumes $$T_{\text{tank}} = T_{\text{chamber}} $$
$$ \frac{d \rho_{\text{tank}}}{dt} = -\frac{\rho_{\text{tank}}}{\tau} $$

$$ \rho_{\text{tank}} = \rho_0 \exp\left(-\frac{t}{\tau}\right) $$

$$ P_{\text{tank}} = P_0 \exp \left (- \frac{t}{\tau} \right) $$

Adiabatic assumption assumes temperature decreases as gas expands in tank

$$\rho_{\text{tank}} = \rho_0 \left[ 1 + \left( \frac{\gamma - 1}{2} \right) \frac{t}{\tau} \right]^{\frac{2}{1-\gamma}}$$
$$P_{\text{tank}} = P_0 \left[ 1 + \left( \frac{\gamma - 1}{2} \right) \frac{t}{\tau} \right]^{\frac{2\gamma}{1-\gamma}}$$
$$T_{\text{tank}} = T_0 \left[ 1 + \left( \frac{\gamma - 1}{2} \right) \frac{t}{\tau} \right]^{-2}$$

The depressurization script keeps iterating until tank reaches 1% of initial mass.

### Monte Carlo
The Monte Carlo function predicts thrust variance using linear hypercube sampling of a design parameter. As opposed to traditional Monte Carlo runs that generate hundreds of samples which may have blankspots or be computationally intensive, linear hypercube sampling generates one sample at even probability intervals. Assuming throat radius is normally distributed it splits the probability distribution in 20 even bins and selects one sample from each bin. The thermal model then outputs the result for each sample and generates a normal distribution curve using the resulting thrust values.

<p align="center">
    <img src="./assets/Monte_Carlo.PNG" width=50%>
</p>

# Getting Started

## Prerequisites
- Python 3.14+
- matplotlib 3.10.6+
- numpy 2.3.3+
- PyQt6 6.9.1+
- PyQt6_sip 13.9.1+
- scipy 1.16.1+

## Installation

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
    python src/main.py
    ```


## Basic Usage
### User Inputs
* All user inputs are assumed in SI units. Changing any option recalculates values instantly. Options not applicable to nozzle type will be **greyed out**.

* **Air, CO2, N2, and Xenon** are available as propellants for analysis.

* **MOC full length nozzle** generates an expansion and straightening section with exit Mach number being used to map out contour.

* **MOC minimum length nozzle** generates a sharper corner at throat with expansion fans canceled out along the contour. It represents the smallest possible length for a nozzle designed by method of characteristics.

* **Conical nozzle option** is based on convergence and divergence angles with respect to throat radius and drawn to inlet and exit radii.

* **Rao Bell Nozzle** is based on throat radius, nozzle length, and area-expansion ratio.

* **Chamber pressure and temperature** set stagnation conditions.

* **Convergence and Divergence angles** are used to generate a conical nozzle.

* **Design Thrust** The thrust optimized for in the gradient descent algorithm. Takes into account momentum and pressure thrust.

* **Optimize Geometry** starts the gradient-descent algorithm iterating nozzle geometries until calculated thrust equals designed thrust.

* **Inlet Dimensions** are used to set geometry properties.

* **Ambient pressure** is used to match exit pressure and determine flow type.

* **Depressurization Type** is used to set isothermal or adiabatic assumption during depressurization.

* **Monte Carlo Option** Generates a Monte Carlo test of the selected property.

* **Run** Turns red when current results are outdated, blue when results are being calculated, and green when results are current.

### Results Summary
* Label prints out the current flow regime.

* Supersonic choked exit pressure - Assumes choked perfectly expanded supersonic flow.

* Subsonic choked exit pressure - Assumes near sonic choked throat that is compressed back to a subsonic exit.

* Exit shock exit pressure - Assumes a normal shock at the very end of the nozzle.

* Initial shock exit pressure - Assumes a normal shock at the initial portion of the divergent section.

* Mass flow rate - Rate of mass transfer through nozzle.

* Exit pressure - Actual exit pressure of the nozzle.

* Propellant mass - Mass of propellant in the nozzle.

* Expansion ratio - Ratio of exit area to throat area.

* Specific impulse - Measure of efficiency a nozzle generates thrust.

* Actual thrust - Thrust generated from $$F=\dot m_e V_e + (p_e-p_{amb})A_e$$

## Sources

[1] J. D. Anderson, Jr., Modern Compressible Flow: With Historical Perspective, 4th ed. New York, NY, USA: McGraw-Hill Education, 2021.

[2] S. A. Whitmore, “Introduction to the method of characteristics and the minimum length nozzle,” MAE 5540 – Propulsion Systems, Utah State University. [Online]. Available: http://mae-nas.eng.usu.edu/MAE_5540_Web/propulsion_systems/section8/section.8.1.pdf

[3] S. Asha, G. Dhatri Naga Mohana, K. Sai Priyanka, and D. Govardhan, "Design of Minimum Length Nozzle Using Method of Characteristics," International Journal of Engineering Research and Technology (IJERT), 2021.

[4] M. A. Khan, S. K. Sardiwal, M. V. S. Sharath, and D. H. Chowdary, "Design of a Supersonic Nozzle using Method of Characteristics," International Journal of Engineering Research & Technology (IJERT), vol. 2, no. 11, Nov. 2013.

## ✉️ Contact  
Adam Matrab — [@am975eng](https://github.com/am975eng)
