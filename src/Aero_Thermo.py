"""
Contains thermodynamic and aerodynamic equations for use in other python scripts.

Useful thermodynamics, fluid mechanics, and aerodynamics expressions are defined here. RS prefix is used for functions that return an equation used by root_scalar to solve for the first input variable. Calc prefix generates a value based on input variables.
"""

import numpy as np


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