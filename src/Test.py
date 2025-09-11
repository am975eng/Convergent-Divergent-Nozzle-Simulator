import numpy as np
from scipy.optimize import root_scalar
import math
from numpy import tan
def A_mach_crit(M, a_ratio):
    # Area-Mach Relation Function
    term = (2 / (k + 1)) * (1 + (k - 1) * (M**2) / 2)
    exponent = (k + 1) / (2 * (k - 1))
    A_mach_crit = (a_ratio) - (1 / M) * (term ** exponent)
    return A_mach_crit

def P_mach(M, p_ratio):
    # P_ratio = P_0/P
    P_mach = (-M)+(((p_ratio**((k-1)/k))-1)**(2/(k-1)))**0.5
    return P_mach

def Area_Mach_x_y(M_y, M_x, A_y, A_x):
    expon = (k+1)/(k-1)
    term_x = 1 + ((k-1)/2)*M_x*M_x
    term_y = 1 + ((k-1)/2)*M_y*M_y
    Area_Mach_x_y = (-A_y/A_x)+(M_x/M_y)*(((term_y/term_x)**expon)**0.5)
    return Area_Mach_x_y

def calc_isen_press(M,P):
    # Uses isentropic relation to calculate pressure at a point using stagnation conditions.
    return P / (((1+ ((k-1)/2) * M * M))**(k/(k-1)))

k = 1.4
R_spec = 287
P_0 = 600000
r_inlet = 0.5
r_throat = 0.05
r_outlet = 0.051
M_sonic = 1

A_star = math.pi*(r_throat**2)
A_inlet = math.pi*(r_inlet**2)
A_outlet = math.pi*(r_outlet**2)
print(A_outlet/A_star)
try:
    M_e_sup = root_scalar(A_mach_crit, bracket=[1,100], args=(A_outlet/A_star)).root
    M_e_sub = root_scalar(A_mach_crit, bracket=[0.0001,1], args=(A_outlet/A_star)).root
    M_e_sup_2 = root_scalar(Area_Mach_x_y, bracket=[1,100], args=(M_sonic, A_outlet, A_star)).root
    M_e_sub_2 = root_scalar(Area_Mach_x_y, bracket=[0.0001,1], args=(M_sonic, A_outlet, A_star)).root
except ValueError as e:
    print("Unable to solve for Mach numbers. Expand solver bracket to ensure solution exists.")

P_e_sup = calc_isen_press(M_e_sup,P_0)
P_e_sub = calc_isen_press(M_e_sub,P_0)


X_T * tan(beta_1) = H - H_m
X_D * tan(theta_1) = H - Y_D
(X_D-X_T)*tan(beta_2-theta_1) = Y_D - H_m
(X_F-X_T) * tan(theta_3) = H_m - Y_F
(X_E-X_D) * tan(mu_D) = Y_D - H_s
(X_F-X_D) * tan(mu_2 + theta_3) = Y_F - Y_D
(X_E - X_F) * tan(theta_3) = (2 + tan(theta_3)*tan(theta_3))*(Y_F-H_s)