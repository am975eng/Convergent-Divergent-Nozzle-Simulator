import math
import numpy as np
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt
"""
Script that solves for minimum length nozzle using method of characteristics 
to generate mach waves or characteristics lines. 

Initially the flow is turned by the wall angle through a Prandtl-Meyer expansion fan from a sonic throat. These mach lines form characteristic lines in which intersection points can be found. By matching characteristic strength we can generate new flow angles and prandtl-meyer angles. Marching along the nozzle the contour is plotted to ensure expansion waves are cancelled at the wall. At the end the flow angle is zero and isentropic straight exit flow is ensured. Contour data is then exported for use in the thruster simulator.
"""

r_throat = .08
r_inlet = .1
r_outlet = .1

def gen_MOC_MLN(M_exit, r_throat, k=1.4, div=7):
    def Prandtl_Meyer(M):
        """
        Calculate Prandtl-Meyer angle.
        
        Args:
            M - Mach number at point of interest
        """
        term_1 = ((k+1)/(k-1))**0.5
        term_2 = (((k-1)/(k+1))*(M**2-1))**0.5
        term_3 = (M**2-1)**0.5
        nu = term_1 * np.arctan(term_2) - np.arctan(term_3)
        return nu
    
    def RS_M_from_nu(M, nu):
        """
        Function called by root_scalar to calculate Mach number using Prandtl-Meyer relation.
        
        Args:
            M - Mach number at point of interest
            nu - Prandtl-Meyer angle"""
        term_1 = ((k+1)/(k-1))**0.5
        term_2 = (((k-1)/(k+1))*(M**2-1))**0.5
        term_3 = (M**2-1)**0.5
        eqn = -nu + term_1 * np.arctan(term_2) - np.arctan(term_3)
        return eqn


    theta_wall_max = Prandtl_Meyer(M_exit)/2    # Wall angle at throat
    delta_theta = 0.375*np.pi/180               # PM Fan Increment
    npts = int(div * (div+1)/2 + div)           # Total number of intersection points
    npts += 1

    # Preallocate arrays
    theta_n = np.zeros(npts)
    nu_n = np.zeros(npts)
    K_minus = np.zeros(npts)
    K_plus = np.zeros(npts)
    M_n = np.zeros(npts)
    mu_n = np.zeros(npts)
    x_n = np.zeros(npts)
    y_n = np.zeros(npts)

    theta_n[0] = theta_wall_max 
    start = 1
    r_throat = 1
    y_n[0] = r_throat
    x_n[0] = 0

    for size in range(div, 0, -1):  # The number of intersecting interior points decreases by 1 with each iteration
        end = start + size - 1
        if start == 1:              # Prandtl-Meyer expansion fan
            for i in range(start,end+1):
                theta_n[i] = delta_theta + (3*np.pi/180) * (i-1) 
                nu_n[i] = theta_n[i]
                K_minus[i] = theta_n[i] + nu_n[i]       # Char line strength constant
                K_plus[i] = theta_n[i] - nu_n[i]
                M_n[i] = root_scalar(RS_M_from_nu, bracket=[1,10], args=(nu_n[i])).root # Local Mach number
                mu_n[i] = np.arcsin(1/M_n[i])
                if i == 1:
                    m = np.tan(theta_n[i] - mu_n[i])
                    x_n[i] = -y_n[0] / m
                    y_n[i] = 0
                    plt.plot([0, x_n[i]], [y_n[0], y_n[i]], 'k-', linewidth=2)
                    plt.annotate(str(i), (x_n[i],y_n[i]))
                else:
                    m_1=np.tan(theta_n[i] - mu_n[i])
                    x_1=x_n[0]
                    y_1=y_n[0]
                    m_2=np.tan(theta_n[i-1] + mu_n[i-1])
                    x_2=x_n[i-1]
                    y_2=y_n[i-1]
                    x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                    y_3 = y_1 + m_1*(x_3 - x_1)
                    x_n[i] = x_3
                    y_n[i] = y_3
                    plt.plot([x_n[0], x_n[i]], [y_n[0], y_n[i]], 'k-', linewidth=2)
                    plt.plot([x_n[i-1], x_n[i]], [y_n[i-1], y_n[i]], 'k-', linewidth=2)
                    plt.annotate(str(i), (x_n[i],y_n[i]))
                
            theta_n[i+1] = theta_n[i]
            nu_n[i+1] = nu_n[i]
            K_minus[i+1] = K_minus[i]
            K_plus[i+1] = K_plus[i]
            M_n[i+1] = M_n[i]
            mu_n[i+1] = mu_n[i]

            theta_start = 0.5*(theta_n[0] + theta_n[i+1])
            m_1 = np.tan(theta_start)
            x_1 = x_n[0]
            y_1 = y_n[0]
            m_2 = np.tan(theta_n[i+1] + mu_n[i+1])
            x_2 = x_n[i]
            y_2 = y_n[i]
            x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
            y_3 = y_1 + m_1*(x_3 - x_1)
            x_n[i+1]=x_3
            y_n[i+1]=y_3
            plt.plot([x_1, x_3], [y_1, y_3], 'k-', linewidth=2)
            plt.plot([x_2, x_3], [y_2, y_3], 'k-', linewidth=2)
            plt.annotate(str(i+1), (x_3,y_3))

        else:
            for i in range(start,end+1):
                if i == start:
                    # Centerline grid point
                    K_minus[i] = K_minus[i-size-1]
                    theta_n[i] = 0
                    K_plus[i] = -K_minus[i]
                    nu_n[i] = 0.5*(K_minus[i] - K_plus[i])
                    M_n[i] = root_scalar(RS_M_from_nu, bracket=[1,10], args=(nu_n[i])).root
                    mu_n[i] = np.arcsin(1/M_n[i])
                    
                    m_1 = np.tan(theta_n[i-size-1] - mu_n[i-size-1])            
                    x_1 = x_n[i-size-1]
                    y_1 = y_n[i-size-1]
                    x_n[i] = x_1 -y_1 / m_1
                    y_n[i] = 0
                    
                    plt.plot([x_1,x_n[i]], [y_1,y_n[i]], 'k-', linewidth=2)
                    plt.annotate(str(i), (x_n[i],y_n[i]))
                else:
                    # Interior intersection grid point
                    K_minus[i] = K_minus[i-size-1]
                    K_plus[i] = K_plus[i-1]
                    theta_n[i] = 0.5*(K_minus[i] + K_plus[i])
                    nu_n[i] = 0.5*(K_minus[i] - K_plus[i])
                    M_n[i] = root_scalar(RS_M_from_nu, bracket=[1,10], args=(nu_n[i])).root
                    mu_n[i] = np.arcsin(1/M_n[i])
                    
                    m_1 = np.tan(theta_n[i-1] + mu_n[i-1])
                    x_1 = x_n[i-1]
                    y_1 = y_n[i-1]
                    m_2 = np.tan(theta_n[i-size-1]  - mu_n[i-size-1] )
                    x_2 = x_n[i-size-1]
                    y_2 = y_n[i-size-1]
                    x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                    y_3 = y_1 + m_1*(x_3 - x_1)
                    x_n[i] = x_3
                    y_n[i] = y_3
                    
                    plt.plot([x_n[i-1], x_n[i]], [y_n[i-1], y_n[i]], 'k-', linewidth=2)
                    plt.plot([x_n[i-size-1], x_n[i]], [y_n[i-size-1], y_n[i]], 'k-', linewidth=2)
                    plt.annotate(str(i), (x_n[i],y_n[i]))

            # Wall Contour
            theta_n[i+1] = theta_n[i]
            nu_n[i+1] = nu_n[i]
            K_minus[i+1] = K_minus[i]
            K_plus[i+1] = K_plus[i]
            M_n[i+1] = M_n[i]
            mu_n[i+1] = mu_n[i]

            theta_start = 0.5*(theta_n[i-size] + theta_n[i+1])
            m_1 = np.tan(theta_start)
            x_1 = x_n[i-size]
            y_1 = y_n[i-size]
            m_2 = np.tan(theta_n[i] + mu_n[i])
            x_2 = x_n[i]
            y_2 = y_n[i]
            x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
            y_3 = y_1 + m_1*(x_3 - x_1)
            x_n[i+1]=x_3
            y_n[i+1]=y_3

            plt.plot([x_1, x_3], [y_1, y_3], 'k-', linewidth=2)
            plt.plot([x_2, x_3], [y_2, y_3], 'k-', linewidth=2)
            plt.annotate(str(i+1), (x_3,y_3))                    

        K_minus[end+1] = K_minus[end]
        K_plus[end+1] = K_plus[end]
        theta_n[end+1] = theta_n[end]
        nu_n[end+1] = nu_n[end]
        M_n[end+1] = M_n[end]
        mu_n[end+1] = mu_n[end]
        
        start = end + 2  # move past block + skip 1

    for i in range(len(theta_n)):
        print(f"i {str(i)} K- {K_minus[i]*180/np.pi:.3f} K+ {K_plus[i]*180/np.pi:.3f} theta {theta_n[i]*180/np.pi:.3f} nu {nu_n[i]*180/np.pi:.3f} M {M_n[i]:.3f} mu {mu_n[i]*180/np.pi:.3f}")

if __name__ == "__main__":
    gen_MOC_MLN(2.4,1,1.4,10)
    plt.axis('equal')
    plt.show()