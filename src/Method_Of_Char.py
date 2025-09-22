import numpy as np
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import Aero_Thermo as AT
"""
Script that solves for minimum length nozzle using method of characteristics to generate mach waves or characteristics lines with a contour nozzle that absorbs expansion waves. 

We assume the flow leaves the nozzle smoothly at zero angle and calculate the properties of the exit Prandtl-Meyer shock as it originates from the throat. This gives us the wall angle at the throat and the first section of contour geometry. A fan of waves is generated from sonic throat conditions to theta_wall_max until it reaches the centerline at which point theta is zero or an intersection point.  Left and right running characteristic lines have constant strength theta - nu and theta + nu respectively which can be set equal to each other at intersection points to solve for flow properties. By marching through this network of points we can solve for all flow properties throughout the nozzle. To generate the wall contour we use the previous wall point and draw a line with angle 0.5(theta_k-1 + theta_k) as well as the left running characteristic line to find the intersection point.
"""

def gen_MOC_MLN(M_exit, r_throat, k=1.4, div=7, print_flag=False):
    """
    Generates a minimum length nozzle using method of characteristics."""
    theta_wall_max = AT.calc_prandtl_meyer(M_exit, k)/2     # Wall angle at throat
    delta_theta = 0.375*np.pi/180                           # PM Fan Increment
    npts = int(div * (div+1)/2 + div)                       # Total number of intersection points
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
    x_contour = []
    y_contour = []

    theta_n[0] = theta_wall_max 
    start = 1
    r_throat = 1
    y_n[0] = r_throat
    x_n[0] = 0
    x_contour.append(x_n[0])
    y_contour.append(y_n[0])

    for size in range(div, 0, -1):  # The number of intersecting interior points decreases by 1 with each iteration
        end = start + size - 1
        if start == 1:              # Prandtl-Meyer expansion fan
            for i in range(start,end+1):
                theta_n[i] = delta_theta + (3*np.pi/180) * (i-1) 
                nu_n[i] = theta_n[i]
                K_minus[i] = theta_n[i] + nu_n[i]       # Char line right running strength constant
                K_plus[i] = theta_n[i] - nu_n[i]        # Char line left running
                M_n[i] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[i], k)).root # Local Mach number
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
            x_contour.append(x_3)
            y_contour.append(y_3)
            plt.plot([x_1, x_3], [y_1, y_3], 'r-', linewidth=2)
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
                    M_n[i] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[i], k)).root
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
                    M_n[i] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[i], k)).root
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
            x_contour.append(x_3)
            y_contour.append(y_3)

            plt.plot([x_1, x_3], [y_1, y_3], 'r-', linewidth=2)
            plt.plot([x_2, x_3], [y_2, y_3], 'k-', linewidth=2)
            plt.annotate(str(i+1), (x_3,y_3))                    

        K_minus[end+1] = K_minus[end]
        K_plus[end+1] = K_plus[end]
        theta_n[end+1] = theta_n[end]
        nu_n[end+1] = nu_n[end]
        M_n[end+1] = M_n[end]
        mu_n[end+1] = mu_n[end]
        
        start = end + 2  # move past block + skip 1

    if print_flag == True:
        for i in range(len(theta_n)):
            print(f"i {str(i)} K- {K_minus[i]*180/np.pi:.3f} K+ {K_plus[i]*180/np.pi:.3f} theta {theta_n[i]*180/np.pi:.3f} nu {nu_n[i]*180/np.pi:.3f} M {M_n[i]:.3f} mu {mu_n[i]*180/np.pi:.3f}")

    return x_contour, y_contour, x_n, y_n, M_n

if __name__ == "__main__":
    x_contour, y_contour, x_n, y_n, M_n = gen_MOC_MLN(2.4,1,1.4,7)
    plt.axis('equal')
    plt.title('Method of Characteristics - Minimum Length Nozzle')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')

    plt.figure()
    x_n = np.append(x_n, max(x_n))
    y_n = np.append(y_n, 0)
    M_n = np.append(M_n, max(M_n))
    x_i = np.linspace(min(x_n), max(x_n), 300)
    y_i = np.linspace(min(y_n), max(y_n), 300)
    X,Y = np.meshgrid(x_i,y_i)
    Z = griddata((x_n, y_n), M_n, (X, Y), method='cubic')
    c = plt.pcolormesh(X, Y, Z, cmap=plt.cm.RdYlBu )
    plt.scatter(x_n, y_n, c=M_n, edgecolor="k", s=1)  # show original points
    plt.plot(x_contour, y_contour, 'k-', linewidth=2)
    plt.title('Mach Number Contour')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.colorbar(c)
    plt.axis('equal')
    plt.show()