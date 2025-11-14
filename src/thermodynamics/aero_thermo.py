"""
Contains thermodynamic and aerodynamic equations for use in other python scripts.

Useful thermodynamics, fluid mechanics, and aerodynamics expressions are defined here. RS prefix is used for functions that return an equation used by root_scalar to solve for the first input variable. Calc prefix generates a value based on input variables.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def RS_Area_Mach_X_Y(M_2, M_x, A_y, A_x, k):
    """
    Function called by root_scalar to calculate Mach number using isentropic area-Mach relation.
    
    Args:
        M_2 - Mach number at y point
        M_x - Mach number at x point
        A_y - Area at y point
        A_x - Area at x point
        k - Ratio of specific heats
    """
    expon = (k+1)/(k-1)
    term_x = 1 + ((k-1)/2)*M_x*M_x
    terM_2 = 1 + ((k-1)/2)*M_2*M_2
    RS_Area_Mach_X_Y = (-A_y/A_x)+(M_x/M_2)*(((terM_2/term_x)**expon)**0.5)
    return RS_Area_Mach_X_Y

def RS_Mach_Press_Isen(M_2, M_1, P_2, P_1, k):
    """
    Function called by root_scalar to calculate Mach number using isentropic pressure-Mach relation.
    
    Args:
        M_2 - Mach number to be calculated
        M_1 - Mach number at point 1
        P_2 - Pressure at point to be calculated
        P_1 - Pressure at x point
        k - Ratio of specific heats
    """
    exp = k/(k-1)
    term_1 = 1 + ((k-1)/2)*M_1*M_1
    term_2 = 1 + ((k-1)/2)*M_2*M_2
    eqn = -(P_2/P_1) + ((term_1/term_2)**exp)
    return eqn

def FS_oblique_angle(vars, M_1, P_1, P_2, k):
    """
    Generates equations for fsolve to solve for beta and theta of oblique shock given initial Mach and pressure before and after shockwave.

    Fsolve unpacks the variables beta, shockwave angle, and theta, flow deflection angle. Using the theta-beta-mach equation and pressure across normal shockwave equation two homogenous equations are generated. 
    
    Args:
        vars - [beta, theta] - Shockwave angle, Flow deflection angle
        M_1 - Mach number before shockwave
        P_1 - Pressure before shockwave
        P_2 - Pressure after shockwave
        k - Ratio of specific heats

    Returns:
        [eqn1, eqn2] - Pressure shockwave equation and theta-beta-mach equation
    """

    beta, theta = vars
    M_1_n = M_1 * np.sin(beta)
    eqn1 = -(P_2/P_1) + (2*k*M_1_n*M_1_n-k+1)/(k+1)
    eqn2 = -np.tan(theta) + (2*(1/np.tan(beta))*(M_1*M_1*np.sin(beta)*np.sin(beta)-1))/(M_1*M_1*(k+np.cos(2*beta))+2)
    return [eqn1, eqn2]

def calc_prandtl_meyer(M, k):
    """
    Calculate Prandtl-Meyer angle.
    
    Args:
        M - Mach number at point of interest
        k - Ratio of specific heats
    """
    term_1 = ((k+1)/(k-1))**0.5
    term_2 = (((k-1)/(k+1))*(M**2-1))**0.5
    term_3 = (M**2-1)**0.5
    nu = term_1 * np.arctan(term_2) - np.arctan(term_3)
    return nu

def RS_mach_prandtl_meyer(M, nu, k):
    """
    Function called by root_scalar to calculate Mach number using Prandtl-Meyer angle.
    
    Args:
        M - Mach number at point of interest
        nu - Prandtl-Meyer angle
        k - Ratio of specific heats
        
    Returns:
        eqn - PM equation"""
    term_1 = ((k+1)/(k-1))**0.5
    term_2 = (((k-1)/(k+1))*(M**2-1))**0.5
    term_3 = (M**2-1)**0.5
    eqn = -nu + term_1 * np.arctan(term_2) - np.arctan(term_3)
    return eqn
        
def calc_isen_press(M,P_0,k):
    """
    Calculate pressure at a point using local mach number and stagnation conditions assuming isentropic calorically perfect flow.
    
    Args:
        M - Mach number at point of interest
        P_0 - Stagnation pressure
        k - Ratio of specific heats
    """
    return P_0 / (((1+ ((k-1)/2) * M * M))**(k/(k-1)))

def calc_isen_stag_press(M,P,k):
    """
    Calculate stagnation pressure at a point using local mach number and pressure assuming isentropic calorically perfect flow. 
    
    Args:
        M - Mach number
        P - Pressure
        k - Ratio of specific heats
    """

    return P * (((1+ ((k-1)/2) * M * M))**(k/(k-1)))

def calc_isen_temp(M,T_0,k):
    """
    Calculate temperature at a point using local mach number and stagnation conditions assuming isentropic calorically perfect flow.
    
    Args:
        M - Mach number at point of interest
        T_0 - Stagnation temperature
        k - Ratio of specific heats
    """
    return T_0 * ((1+ ((k-1)/2)*M*M)**-1)

def calc_M_P_normal(M_1, P_1, k):
    """
    Calculate Mach number and static pressure after a normal shock
    
    Args:
        M_1 - Mach number before shockwave
        P_1 - Pressure before shockwave
        k - Ratio of specific heats
    """
    M_2 = ((M_1**2 + (2/(k-1)))/((2*k/(k-1))*M_1*M_1-1))**0.5                    
    P_2 = P_1*(((2*k/(k+1))*M_1*M_1) - (k-1)/(k+1))                               
    return M_2, P_2

def calc_ada_dPdT(P_0_init, tau, t, k):
    dPdt = - (k * P_0_init / tau) * (1 + ((k - 1) / 2) * (t / tau))**((2 * k) / (1 - k) - 1)
    return dPdt

def calc_isotherm_dPdt(P_0_init, tau, t):
    dPdt = - (P_0_init/tau) * np.exp(-t/tau)
    return dPdt

def calc_ada_drhodt(rho_0_init, tau, t, k):
    drhodt = - (rho_0_init / tau) * ((1 + ((k-1)/2) * (t/tau))**((1 + k) / (1 - k)))
    return drhodt

def calc_isotherm_drhodt(rho_0_init, tau, t):
    drhodt = - (rho_0_init / tau) * np.exp(-t/tau)
    return drhodt

def calc_ada_dTdt(T_0_init, tau, t, k):
    dTdt = - T_0_init * ((k-1)/tau) * ((1 + ((k-1)/2) * (t/tau))**-3)
    return dTdt

def calc_rao_para_ang(area_ratio_outlet, len_per = 85):
    theta_n = {60:[4.371, 27.255,
                5.817, 29.185,
                8.341, 31.244,
                13.188, 33.313,
                20.977, 35.076,
                32.989, 36.715,
                55.585, 38.361,
                80.685, 39.563,
                ],70:[4.613, 24.745,
                6.540, 26.619,
                9.221, 28.309,
                13.613, 29.881,
                19.985, 31.268,
                31.071, 32.844,
                54.808, 34.678,
                84.737, 36.070,
                ],80:[4.415, 22.409,
                6.187, 24.283,
                9.239, 25.978,
                14.363, 27.677,
                23.515, 29.320,
                38.060, 30.901,
                61.965, 32.359,
                86.426, 33.189],
                90:[4.682, 20.881,
                6.452, 22.446,
                9.633, 24.264,
                15.859, 26.092,
                25.372, 27.855,
                42.989, 29.685,
                74.534, 31.518,
                ]}
    
    theta_e = {60:[4.347, 20.199,
6.079, 18.146,
9.422, 16.532,
15.735, 15.171,
26.725, 14.241,
42.603, 13.488,
69.886, 12.861,
],
                   70:[3.885, 17.305,
5.622, 15.439,
8.516, 13.884,
19.972, 12.004,
32.393, 11.192,
50.170, 10.559,
77.243, 10.110,
],
                   80:[3.829, 14.236,
5.702, 12.312,
8.736, 10.881,
14.420, 9.764,
24.211, 8.832,
39.494, 8.021,
63.675, 7.454,
],
90:[4.510, 10.878,
6.832, 9.262,
11.150, 8.021,
20.882, 7.039,
34.448, 6.535,
52.423, 6.269,
78.862, 6.063]}
    def exponential_func(x, a, b, c, d):
        return a * x + b/x + c*(x**0.5) + d

    x = np.linspace(4, 85, 100)
    for key in theta_n:
        x_data = theta_n[key][::2]
        y_data = theta_n[key][1::2]
        plt.plot(x_data, y_data, 'r', linewidth=2)
        popt, pcov = curve_fit(exponential_func, x_data, y_data)
        a, b, c, d = popt
        #y = exponential_func(x, a, b, c)
        y = exponential_func(x, a, b, c, d)
        plt.plot(x, y, 'b--', linewidth=2)

    for key in theta_e:
        x_data = theta_e[key][::2]
        y_data = theta_e[key][1::2]
        plt.plot(x_data, y_data, 'r', linewidth=2)
        popt, pcov = curve_fit(exponential_func, x_data, y_data)
        a, b, c, d = popt
        #y = exponential_func(x, a, b, c)
        y = exponential_func(x, a, b, c, d)
        plt.plot(x, y, 'b--', linewidth=2)


        
    plt.show()



calc_rao_para_ang(5)