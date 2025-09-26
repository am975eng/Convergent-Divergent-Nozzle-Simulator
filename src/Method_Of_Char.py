import numpy as np
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import Aero_Thermo as AT
"""
Script that solves for minimum length nozzle using method of characteristics to generate mach waves or characteristics lines with a contour nozzle that absorbs expansion waves. 

We assume the flow leaves the nozzle smoothly at zero angle and calculate the properties of the exit Prandtl-Meyer shock as it originates from the throat. This gives us the wall angle at the throat and the first section of contour geometry. A fan of waves is generated from sonic throat conditions to theta_wall_max until it reaches the centerline at which point theta is zero or an intersection point.  Left and right running characteristic lines have constant strength theta - nu and theta + nu respectively which can be set equal to each other at intersection points to solve for flow properties. By marching through this network of points we can solve for all flow properties throughout the nozzle. To generate the wall contour we use the previous wall point and draw a line with angle 0.5(theta_k-1 + theta_k) as well as the left running characteristic line to find the intersection point.
"""

def gen_MOC_MLN(M_exit, r_throat, k=1.4, div=7, print_flag=False, plot_flag=False):
    """
    Generates a minimum length nozzle using method of characteristics.

    Args:
        M_exit (float): Exit Mach number.
        r_throat (float): Throat radius.
        k (float, optional): Ratio of specific heats. Defaults to 1.4.
        div (int, optional): Number of divisions. Defaults to 7.
        print_flag (bool, optional): Print MOC results in a table. Defaults to False.
        plot_flag (bool, optional): Plot MOC results. Defaults to False.

    Returns:
        x_contour (list): Contour x coordinates.
        y_contour (list): Contour y coordinates.
        M_n (numpy.ndarray): Mach numbers."""
    theta_wall_max = AT.calc_prandtl_meyer(M_exit, k)/2     # Wall angle at throat
    init_theta = 0.375*np.pi/180                           # PM Fan Increment
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
    delta_theta = (theta_wall_max - init_theta)/(div-1)
    y_n[0] = r_throat
    x_n[0] = 0
    x_contour.append(x_n[0])
    y_contour.append(y_n[0])

    
    idx = 1                                         # Current point index                     
    row = 1                                         # Vertical row index
    while idx <= npts-1:
        row_len = div - row + 2
        for row_index in range(row_len):            # Iterate through each row
            if row_index==0:                        # Centerline Point
                if idx == 1:                        # First PM centerline point
                    theta_n[idx] = init_theta + delta_theta * (idx-1) 
                    nu_n[idx] = theta_n[idx]
                    K_minus[idx] = theta_n[idx] + nu_n[idx]       # Char line right running strength constant
                    K_plus[idx] = theta_n[idx] - nu_n[idx]        # Char line left running
                    M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[idx], k)).root # Local Mach number
                    mu_n[idx] = np.arcsin(1/M_n[idx])
                    m = np.tan(theta_n[idx] - mu_n[idx])
                    x_n[idx] = -y_n[0] / m
                    y_n[idx] = 0
                    if plot_flag:
                        plt.plot([x_n[0], x_n[idx]], [y_n[0], y_n[idx]], 'k-', linewidth=2)
                        plt.annotate(str(idx), (x_n[idx],y_n[idx]))
                else:                               # Other centerline points
                    K_minus[idx] = K_minus[idx-row_len]
                    theta_n[idx] = 0
                    K_plus[idx] = -K_minus[idx]
                    nu_n[idx] = 0.5*(K_minus[idx] - K_plus[idx])
                    M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[idx], k)).root
                    mu_n[idx] = np.arcsin(1/M_n[idx])
                    
                    m_1 = np.tan(theta_n[idx-row_len] - mu_n[idx-row_len])            
                    x_1 = x_n[idx-row_len]
                    y_1 = y_n[idx-row_len]
                    x_n[idx] = x_1 -y_1 / m_1
                    y_n[idx] = 0
                    
                    if plot_flag:
                        plt.plot([x_1,x_n[idx]], [y_1,y_n[idx]], 'k-', linewidth=2)
                        plt.annotate(str(idx), (x_n[idx],y_n[idx]))
            elif row_index == row_len-1:            # Wall Point
                theta_n[idx] = theta_n[idx-1]
                nu_n[idx] = nu_n[idx-1]
                K_minus[idx] = K_minus[idx-1]
                K_plus[idx] = K_plus[idx-1]
                M_n[idx] = M_n[idx-1]
                mu_n[idx] = mu_n[idx-1]

                theta_start = 0.5*(theta_n[idx-row_len] + theta_n[idx])
                m_1 = np.tan(theta_start)
                x_1 = x_n[idx-row_len]
                y_1 = y_n[idx-row_len]
                m_2 = np.tan(theta_n[idx] + mu_n[idx])
                x_2 = x_n[idx-1]
                y_2 = y_n[idx-1]
                x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                y_3 = y_1 + m_1*(x_3 - x_1)
                x_n[idx]=x_3
                y_n[idx]=y_3
                x_contour.append(x_3)
                y_contour.append(y_3)

                if plot_flag:
                    plt.plot([x_1, x_3], [y_1, y_3], 'r-', linewidth=2)
                    plt.plot([x_2, x_3], [y_2, y_3], 'k-', linewidth=2)
                    plt.annotate(str(idx), (x_3,y_3))
            else:                                   # Interior grid intersection point
                if row == 1:                        # PM fan interior grid points
                    theta_n[idx] = init_theta + delta_theta * (idx-1) 
                    nu_n[idx] = theta_n[idx]
                    K_minus[idx] = theta_n[idx] + nu_n[idx]       # Char line right running strength constant
                    K_plus[idx] = theta_n[idx] - nu_n[idx]        # Char line left running
                    M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[idx], k)).root # Local Mach number
                    mu_n[idx] = np.arcsin(1/M_n[idx])
                
                    m_1=np.tan(theta_n[idx] - mu_n[idx])
                    x_1=x_n[0]
                    y_1=y_n[0]
                    m_2=np.tan(theta_n[idx-1] + mu_n[idx-1])
                    x_2=x_n[idx-1]
                    y_2=y_n[idx-1]
                    x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                    y_3 = y_1 + m_1*(x_3 - x_1)
                    x_n[idx] = x_3
                    y_n[idx] = y_3
                    if plot_flag:
                        plt.plot([x_n[0], x_n[idx]], [y_n[0], y_n[idx]], 'k-', linewidth=2)
                        plt.plot([x_n[idx-1], x_n[idx]], [y_n[idx-1], y_n[idx]], 'k-', linewidth=2)
                        plt.annotate(str(idx), (x_n[idx],y_n[idx]))
                
                else:                               # Interior grid points
                    K_minus[idx] = K_minus[idx-row_len]
                    K_plus[idx] = K_plus[idx-1]
                    theta_n[idx] = 0.5*(K_minus[idx] + K_plus[idx])
                    nu_n[idx] = 0.5*(K_minus[idx] - K_plus[idx])
                    M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[idx], k)).root
                    mu_n[idx] = np.arcsin(1/M_n[idx])
                    
                    m_1 = np.tan(theta_n[idx-1] + mu_n[idx-1])
                    x_1 = x_n[idx-1]
                    y_1 = y_n[idx-1]
                    m_2 = np.tan(theta_n[idx-row_len]  - mu_n[idx-row_len] )
                    x_2 = x_n[idx-row_len]
                    y_2 = y_n[idx-row_len]
                    x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                    y_3 = y_1 + m_1*(x_3 - x_1)
                    x_n[idx] = x_3
                    y_n[idx] = y_3
                    
                    if plot_flag:
                        plt.plot([x_n[idx-1], x_n[idx]], [y_n[idx-1], y_n[idx]], 'k-', linewidth=2)
                        plt.plot([x_n[idx-row_len], x_n[idx]], [y_n[idx-row_len], y_n[idx]], 'k-', linewidth=2)
                        plt.annotate(str(idx), (x_n[idx],y_n[idx]))                      
            idx +=1
        row += 1

    if print_flag == True:
        for i in range(len(theta_n)):
            print(f"i {str(i)} K- {K_minus[i]*180/np.pi:.3f} K+ {K_plus[i]*180/np.pi:.3f} theta {theta_n[i]*180/np.pi:.3f} nu {nu_n[i]*180/np.pi:.3f} M {M_n[i]:.3f} mu {mu_n[i]*180/np.pi:.3f} x {x_n[i]:.3f} y {y_n[i]:.3f}")

    return x_contour, y_contour, x_n, y_n, M_n

def gen_MOC_FLN(M_exit, r_throat, k=1.4, div=7, print_flag=False, plot_flag=False):
    """
    Generates a divergent nozzle contour using method of characteristics with an expansion and straightening section.
    """
    
    # Expansion Section based on circular arc
    res = 150                                               # Number of points to use for each contour
    theta_wall_max = AT.calc_prandtl_meyer(M_exit, k)/2     # Maximum wall angle
    theta_final_expand = -(np.pi/2-theta_wall_max)          # Final angle of arc
    theta_expand = np.linspace(-np.pi/2, theta_final_expand, res)   # Theta for circular arc
    scale_factor = 5
    r_expand = r_throat * scale_factor                      # Radius of arc

    theta_expand = np.linspace(0,theta_wall_max, res)
    x_expand = r_expand * np.sin(theta_expand)
    y_expand = r_throat + r_expand * (1-np.cos(theta_expand))

    start = 1
    npts = int(div*(div+1)/2 + 2*div)
    theta_n = np.zeros(npts)
    theta_n[0:div+1] = np.linspace(0,theta_wall_max, div+1)
    nu_n = np.zeros(npts)
    M_n = np.zeros(npts)
    mu_n = np.zeros(npts)
    K_minus = np.zeros(npts)
    K_plus = np.zeros(npts)
    x_n = np.zeros(npts)
    y_n = np.zeros(npts)
    x_n[0] = 0
    y_n[0] = r_throat

    idx = 1                                         # Current point index                     
    row_len = div

    while idx <= npts:
        for row_index in range(row_len):            # Iterate through each row
            if idx <= div:                          # Expansion section wall points
                nu_n[idx] = theta_n[idx]
                K_minus[idx] = theta_n[idx] + nu_n[idx]       # Char line right running strength constant
                K_plus[idx] = theta_n[idx] - nu_n[idx]        # Char line left running
                M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[idx], k)).root 
                mu_n[idx] = np.arcsin(1/M_n[idx])
                
                x_n[idx] = r_expand * np.sin(theta_n[idx])
                y_n[idx] = r_throat + r_expand * (1-np.cos(theta_n[idx]))
                row_len = div + 2
                if plot_flag:
                    plt.scatter(x_n[idx], y_n[idx], s=1)
                    plt.annotate(str(idx), (x_n[idx],y_n[idx]))

            elif row_index == 0:                    # Centerline intercept
                print(f"Center Line idx {idx}")
                K_minus[idx] = K_minus[idx-div]
                K_plus[idx] = -K_minus[idx]
                theta_n[idx] = 0
                nu_n[idx] = K_minus[idx]
                M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[idx], k)).root 
                mu_n[idx] = np.arcsin(1/M_n[idx])

                slope = (theta_n[idx-div] - mu_n[idx-div] - mu_n[idx])/2
                x_n[idx] = - (y_n[idx-div] / np.tan(slope)) + x_n[idx-div]
                y_n[idx] = 0

                if plot_flag:
                    plt.scatter(x_n[idx], y_n[idx], s=1)
                    plt.annotate(str(idx), (x_n[idx],y_n[idx]))
            elif row_index == row_len-1:      # Wall points
                print(f"Wall idx {idx}")
            else:                               # Interior grid points
                #print(f"idx {idx}")
                #print(f"idx - row index {idx - row_len}")
                K_minus[idx] = K_minus[idx-row_len]
                K_plus[idx] = K_plus[idx-1]
                theta_n[idx] = (K_minus[idx] + K_plus[idx])/2
                nu_n[idx] = (K_minus[idx] - K_plus[idx])/2
                M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10], args=(nu_n[idx], k)).root
                mu_n[idx] = np.arcsin(1/M_n[idx])

                m_1 = np.tan(theta_n[idx-1]+mu_n[idx-1])
                x_1 = x_n[idx-1]
                y_1 = y_n[idx-1]
                m_2 = np.tan(theta_n[idx-row_len] - mu_n[idx-row_len])
                x_2 = x_n[idx-row_len]
                y_2 = y_n[idx-row_len]
                x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                y_3 = y_1 + m_1*(x_3 - x_1)
                x_n[idx] = x_3
                y_n[idx] = y_3

                if plot_flag:
                    plt.scatter(x_n[idx], y_n[idx], s=1)
                    plt.annotate(str(idx), (x_n[idx],y_n[idx]))

            idx += 1
        row_len -= 1

    for i in range(len(theta_n)):
        print(f" i {i} theta_n {theta_n[i]*180/np.pi:.3f} nu_n {nu_n[i]*180/np.pi:.3f} K- {K_minus[i]*180/np.pi:.3f} K+ {K_plus[i]*180/np.pi:.3f} M_n {M_n[i]:.3f} mu_n {mu_n[i]*180/np.pi:.3f} x_n {x_n[i]:.3f} y_n {y_n[i]:.3f}")
    
    # plt.plot(x_expand, y_expand, 'k-')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # x_contour, y_contour, x_n, y_n, M_n = gen_MOC_MLN(2.4,1,1.4,7, True, True)
    # plt.axis('equal')
    # plt.title('Method of Characteristics - Minimum Length Nozzle')
    # plt.xlabel('X Position (m)')
    # plt.ylabel('Y Position (m)')

    # plt.figure()
    # x_n = np.append(x_n, max(x_n))
    # y_n = np.append(y_n, 0)
    # M_n = np.append(M_n, max(M_n))
    # x_i = np.linspace(min(x_n), max(x_n), 300)
    # y_i = np.linspace(min(y_n), max(y_n), 300)
    # X,Y = np.meshgrid(x_i,y_i)
    # Z = griddata((x_n, y_n), M_n, (X, Y), method='cubic')
    # c = plt.pcolormesh(X, Y, Z, cmap=plt.cm.RdYlBu )
    # plt.scatter(x_n, y_n, c=M_n, edgecolor="k", s=1)  # show original points
    # plt.plot(x_contour, y_contour, 'k-', linewidth=2)
    # plt.title('Mach Number Contour')
    # plt.xlabel('X Position (m)')
    # plt.ylabel('Y Position (m)')
    # plt.colorbar(c)
    # plt.axis('equal')
    # plt.show()

    gen_MOC_FLN(2.4,1,1.4,7, True, True)