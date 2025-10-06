"""
Script that generates a nozzle contour by solving for supersonic flow field 
using the method of characteristics.

gen_MOC_MLN generates a minimum length nozzle contour by solving for the Mach
line at exit given an exit Mach number and assuming straight flow. Following the
Mach line to the throat after reflection over the centerline gives a wall angle
for the throat. At the throat a Prandtl-Meyer expansion fan is generated up to 
the wall angle with theta and PM angle being set equal. K minus and K plus 
represent characteristic line strengths for right and left running lines
respectively. Mach number is calculated based on an inverse Prandtl-Meyer 
equation and mu is generated from the formula for Mach line angle. We loop 
through checking whether a point is on the centerline, wall, or interior grid 
and solve for fluid properties based on type. As we march along the contour is
mapped until reaching the exit.

gen_MOC_FLN operates similarly to gen_MOC_MLN but with an expansion and
straightening section as opposed to only a straightening section. The expansion
section avoids the sharp turn of the MLN and prevents flow separation. The
expansion section is based on a circular arc that increases in angle from 0 to
theta max based on M_exit. A fan of characteristic lines is generated from the 
expansion section with centerline, wall, and intersection points being solved
similarily.
"""

import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import Aero_Thermo as AT


def high_poly_law(x, a, b, c, d, e):
    return a * (x**4) + b * (x**3) + c * (x**2) + d * x + e

def gen_MOC_MLN(M_exit, r_throat, k=1.4, div=7, print_flag=False,
                plot_flag=False):
    """
    Generates a minimum length nozzle using method of characteristics.

    Args:
        M_exit (float) - Exit Mach number.
        r_throat (float) - Throat radius.
        k (float, optional) - Ratio of specific heats. Defaults to 1.4.
        div (int, optional) - Number of initial grid lines. Defaults to 7.
        print_flag (bool, optional) - Print MOC results. Defaults False
        plot_flag (bool, optional) - Plot MOC results. Defaults False.

    Returns:
        x_contour (list) - Contour x coordinates.
        y_contour (list) - Contour y coordinates.
        x_n (numpy.ndarray) - X coordinates.
        y_n (numpy.ndarray) - Y coordinates.
        M_n (numpy.ndarray) - Mach numbers."""

    theta_wall_max = AT.calc_prandtl_meyer(M_exit, k)/2 # Wall angle at throat
    init_theta = 0.375*np.pi/180                        # PM Fan Initial Ang
    npts = int(div * (div+1)/2 + div) + 1               # Total number of points

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
    col = np.zeros(npts)

    res = 150                                           # No. of contour points
    theta_n[0] = theta_wall_max                         # Wall angle
    delta_theta = (theta_wall_max - init_theta)/(div-1)
    y_n[0] = r_throat
    x_n[0] = 0
    x_contour.append(x_n[0])
    y_contour.append(y_n[0])

    if plot_flag:
        plt.figure()
        plt.axis('equal')
        plt.title('Method of Characteristics - Minimum Length Nozzle')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        colors = ['red', 'yellow', 'green']
        n_colors = div  # Number of colors you want
        custom_cmap = LinearSegmentedColormap.from_list('red_green',
                                                        colors, N=n_colors)

    idx = 1                                         # Current point index
    row = 1                                         # Vertical row index
    while idx <= npts-1:
        row_len = div - row + 2                     # Length of vertical row
        for row_index in range(row_len):            # Iterate through each row
            if row_index==0:                        # Centerline Point
                if idx == 1:                        # First PM centerline point
                    theta_n[idx] = init_theta + delta_theta * (idx-1)
                    nu_n[idx] = theta_n[idx]        # PM angle
                    K_minus[idx] = theta_n[idx] + nu_n[idx] # Char line strength
                    K_plus[idx] = theta_n[idx] - nu_n[idx]  # Left char line
                    M_n[idx] = root_scalar(         # Local Mach number
                        AT.RS_mach_prandtl_meyer, bracket=[1,10],
                        args=(nu_n[idx], k)).root
                    mu_n[idx] = np.arcsin(1/M_n[idx])   # Char line angle
                    col[idx] = row_index

                    m = np.tan(theta_n[idx] - mu_n[idx])
                    x_n[idx] = -y_n[0] / m
                    y_n[idx] = 0
                    if plot_flag:
                        plt.plot([x_n[0], x_n[idx]], [y_n[0], y_n[idx]],
                                 c=custom_cmap(int(col[idx])), linewidth=2)
                        plt.annotate(str(idx), (x_n[idx],y_n[idx]))
                else:                               # Other centerline points
                    K_minus[idx] = K_minus[idx-row_len]
                    theta_n[idx] = 0
                    K_plus[idx] = -K_minus[idx]
                    nu_n[idx] = 0.5*(K_minus[idx] - K_plus[idx])
                    M_n[idx] = root_scalar(
                        AT.RS_mach_prandtl_meyer, bracket=[1,10],
                        args=(nu_n[idx], k)).root
                    mu_n[idx] = np.arcsin(1/M_n[idx])
                    col[idx] = col[idx-row_len]

                    m_1 = np.tan(
                        (theta_n[idx-row_len] - mu_n[idx-row_len] - mu_n[idx])
                        /2)
                    x_1 = x_n[idx-row_len]
                    y_1 = y_n[idx-row_len]
                    x_n[idx] = x_1 - y_1 / m_1
                    y_n[idx] = 0

                    if plot_flag:
                        plt.plot([x_1,x_n[idx]], [y_1,y_n[idx]],
                                 c=custom_cmap(int(col[idx])), linewidth=2)
                        plt.annotate(str(idx), (x_n[idx],y_n[idx]))
            elif row_index == row_len-1:            # Wall Point
                theta_n[idx] = theta_n[idx-1]       # Flow based on prev
                nu_n[idx] = nu_n[idx-1]
                K_minus[idx] = K_minus[idx-1]
                K_plus[idx] = K_plus[idx-1]
                M_n[idx] = M_n[idx-1]
                mu_n[idx] = mu_n[idx-1]

                # Calculate intersection of char line and previous wall point
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
                    plt.plot([x_1, x_3], [y_1, y_3], 'k-', linewidth=2)
                    plt.plot([x_2, x_3], [y_2, y_3],
                             c=custom_cmap(int(col[row])), linewidth=2)
                    plt.annotate(str(idx), (x_3,y_3))
            else:                               # Interior intersection points
                if row == 1:                    # PM fan interior points
                    theta_n[idx] = init_theta + delta_theta * (idx-1)
                    nu_n[idx] = theta_n[idx]
                    K_minus[idx] = theta_n[idx] + nu_n[idx] # Right ch strength
                    K_plus[idx] = theta_n[idx] - nu_n[idx]  # Left char strength
                    M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, # Mach
                                           bracket=[1,10],
                                           args=(nu_n[idx], k)).root
                    mu_n[idx] = np.arcsin(1/M_n[idx])
                    col[idx] = idx

                    # Throat Point
                    m_1=np.tan(theta_n[idx] - mu_n[idx])
                    x_1=x_n[0]
                    y_1=y_n[0]

                    # Prev point on K plus char line
                    m_2=np.tan(theta_n[idx-1] + mu_n[idx-1])
                    x_2=x_n[idx-1]
                    y_2=y_n[idx-1]

                    # Intersection point
                    x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                    y_3 = y_1 + m_1*(x_3 - x_1)
                    x_n[idx] = x_3
                    y_n[idx] = y_3
                    if plot_flag:
                        plt.plot([x_n[0], x_n[idx]], [y_n[0], y_n[idx]],
                                 c=custom_cmap(int(col[idx])), linewidth=2)
                        plt.plot([x_n[idx-1], x_n[idx]], [y_n[idx-1], y_n[idx]],
                                 c=custom_cmap(int(col[row])), linewidth=2)
                        plt.annotate(str(idx), (x_n[idx],y_n[idx]))

                else:                                   # Interior grid points
                    K_minus[idx] = K_minus[idx-row_len] # K minus from prev row
                    K_plus[idx] = K_plus[idx-1]         # K plus from prev point
                    theta_n[idx] = 0.5*(K_minus[idx] + K_plus[idx])
                    nu_n[idx] = 0.5*(K_minus[idx] - K_plus[idx])
                    M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer,
                                           bracket=[1,10],
                                           args=(nu_n[idx], k)).root
                    mu_n[idx] = np.arcsin(1/M_n[idx])
                    col[idx] = col[idx-row_len]

                    m_1 = np.tan((theta_n[idx-1] + mu_n[idx-1] + theta_n[idx] +
                                  mu_n[idx])/2)
                    x_1 = x_n[idx-1]
                    y_1 = y_n[idx-1]
                    m_2 = np.tan((theta_n[idx-row_len] - mu_n[idx-row_len] +
                                  theta_n[idx] - mu_n[idx])/2)
                    x_2 = x_n[idx-row_len]
                    y_2 = y_n[idx-row_len]
                    x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                    y_3 = y_1 + m_1*(x_3 - x_1)
                    x_n[idx] = x_3
                    y_n[idx] = y_3

                    if plot_flag:
                        plt.plot([x_n[idx-1], x_n[idx]], [y_n[idx-1], y_n[idx]],
                                 c=custom_cmap(int(col[row])), linewidth=2)
                        plt.plot([x_n[idx-row_len], x_n[idx]],
                                 [y_n[idx-row_len], y_n[idx]],
                                 c=custom_cmap(int(col[idx])), linewidth=2)
                        plt.annotate(str(idx), (x_n[idx],y_n[idx]))
            idx +=1
        row += 1

    if print_flag:
        for i in range(len(theta_n)):
            print(f"i {str(i)} K- {K_minus[i]*180/np.pi:.3f}" +
                  f"K+ {K_plus[i]*180/np.pi:.3f}" +
                  f"theta {theta_n[i]*180/np.pi:.3f}" +
                  f" nu {nu_n[i]*180/np.pi:.3f} M {M_n[i]:.3f}" +
                  f"mu {mu_n[i]*180/np.pi:.3f} x {x_n[i]:.3f} y {y_n[i]:.3f}")
            
    # Fit nonlinear regression to MOC contour
    p0 = [1.0, 1.0, 1.0, 1.0, 1.0]
    popt, pcov = curve_fit(high_poly_law, x_contour, y_contour, p0=p0,
                           maxfev=10000)
    a, b, c, d, e = popt

    # Generate contour using fitted coefficients
    x_contour_fit = np.linspace(x_contour[0], x_contour[-1], res)
    y_contour_fit = high_poly_law(x_contour_fit, a, b, c, d, e)

    # Curve fitting can yield points slightly smalller than throat
    offset = y_contour[0] - y_contour_fit[0]
    if offset > 0:
        y_contour_fit = y_contour_fit + offset*1.01

    return x_contour_fit, y_contour_fit, x_n, y_n, M_n

def gen_MOC_FLN(M_exit, r_throat, k=1.4, div=7,
                print_flag=False, plot_flag=False):
    """
    Generates a divergent nozzle contour using method of characteristics with an
    expansion and straightening section.

    Args:
        M_exit (float) - Exit Mach number
        r_throat (float) - Throat radius
        k (float, optional) - Specific heat ratio. Defaults to 1.4.
        div (int, optional) - Number of divisions. Defaults to 7.
        print_flag (bool, optional) - Print iteration details. Defaults to False.
        plot_flag (bool, optional) - Plot contour. Defaults to False.

    Returns:
        x_contour (np.array) - Contour x positions
        y_contour (np.array) - Contour y positions
        x_n (np.array) - Point x positions
        y_n (np.array) - Point y positions
        M_n (np.array) - Point Mach numbers
    """

    # Expansion Section based on circular arc
    res = 150                                               # No. contour points
    theta_wall_max = AT.calc_prandtl_meyer(M_exit, k)/2     # Maximum wall angle
    scale_factor = 1
    r_expand = r_throat * scale_factor                      # Radius of arc
    theta_expand = np.linspace(0,theta_wall_max, res)
    x_expand = r_expand * np.sin(theta_expand)
    y_expand = r_throat + r_expand * (1-np.cos(theta_expand))

    # Preallocation
    npts = int(div*(div+1)/2 + 2*div + 1)
    theta_n = np.zeros(npts)
    theta_n[0:div+1] = np.linspace(0,theta_wall_max, div+1)
    nu_n = np.zeros(npts)
    M_n = np.zeros(npts)
    mu_n = np.zeros(npts)
    K_minus = np.zeros(npts)
    K_plus = np.zeros(npts)
    x_n = np.zeros(npts)
    y_n = np.zeros(npts)
    col = np.zeros(npts)
    x_str = []
    y_str = []

    x_n[0] = 0
    y_n[0] = r_throat
    idx = 1                                         # Current point index
    # List of row lengths
    grid_row_pts = [div, div+1] + [div-i for i in range(div-1)]

    if plot_flag:
        plt.figure()
        plt.axis('equal')
        plt.title('Method of Characteristics - Full Length Nozzle')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        colors = ['red', 'yellow', 'green']
        n_colors = div  # Number of colors you want
        custom_cmap = LinearSegmentedColormap.from_list('red_green', colors,
                                                        N=n_colors)

    for row_num, row_len in enumerate(grid_row_pts):    # Loop row lengths
        for row_index in range(row_len):                # Loop each row
            if idx <= div:                              # Expansion section
                nu_n[idx] = theta_n[idx]                # PM angle
                K_minus[idx] = theta_n[idx] + nu_n[idx] # Char str right
                K_plus[idx] = theta_n[idx] - nu_n[idx]  # Char str left
                M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, # Mach No.
                                       bracket=[1,10], args=(nu_n[idx], k)).root
                mu_n[idx] = np.arcsin(1/M_n[idx])       # Shock angle
                col[idx] = row_index                    # Color

                x_n[idx] = r_expand * np.sin(theta_n[idx])
                y_n[idx] = r_throat + r_expand * (1-np.cos(theta_n[idx]))
                if plot_flag:
                    plt.plot([x_n[idx-1], x_n[idx]], [y_n[idx-1], y_n[idx]],
                             "k-", linewidth=1)
                    plt.scatter(x_n[idx], y_n[idx], s=1)
                    plt.annotate(str(idx), (x_n[idx],y_n[idx]))

            elif row_index == 0:                        # Centerline intercept
                if idx <= div * 2:
                    offset = grid_row_pts[row_num-1]
                else:
                    offset = grid_row_pts[row_num]

                K_minus[idx] = K_minus[idx-offset]
                K_plus[idx] = -K_minus[idx]
                theta_n[idx] = 0
                nu_n[idx] = K_minus[idx]
                M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10],
                                       args=(nu_n[idx], k)).root
                mu_n[idx] = np.arcsin(1/M_n[idx])
                col[idx] = col[idx-offset]

                # Solve for slope and intercept
                m1 = np.tan(
                    (theta_n[idx-offset] - mu_n[idx-offset] - mu_n[idx])/2)
                x_n[idx] = - (y_n[idx-offset] / m1) + x_n[idx-offset]
                y_n[idx] = 0

                if plot_flag:
                    plt.plot([x_n[idx-offset], x_n[idx]],
                             [y_n[idx-offset], y_n[idx]],
                             c=custom_cmap(int(col[idx])), linewidth=1)
                    plt.scatter(x_n[idx], y_n[idx], s=1)
                    plt.annotate(str(idx), (x_n[idx],y_n[idx]))
            elif row_index == row_len-1:      # Wall points
                offset = grid_row_pts[row_num]
                theta_n[idx] = theta_n[idx-1]
                nu_n[idx] = nu_n[idx-1]
                K_minus[idx] = K_minus[idx-1]
                K_plus[idx] = K_plus[idx-1]
                M_n[idx] = M_n[idx-1]
                mu_n[idx] = mu_n[idx-1]

                theta_start = 0.5*(theta_n[idx-offset] + theta_n[idx])
                m_1 = np.tan(theta_start)
                x_1 = x_n[idx-offset]
                y_1 = y_n[idx-offset]
                m_2 = np.tan(theta_n[idx] + mu_n[idx])
                x_2 = x_n[idx-1]
                y_2 = y_n[idx-1]
                x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                y_3 = y_1 + m_1*(x_3 - x_1)
                x_n[idx]=x_3
                y_n[idx]=y_3
                x_str.append(x_3)
                y_str.append(y_3)

                if plot_flag:
                    plt.plot([x_n[idx-1], x_n[idx]], [y_n[idx-1], y_n[idx]],
                             c=custom_cmap(int(col[row_num])), linewidth=1)
                    plt.plot([x_n[idx-offset], x_n[idx]],
                             [y_n[idx-offset], y_n[idx]], "k-", linewidth=1)
                    plt.scatter(x_n[idx], y_n[idx], s=1)
                    plt.annotate(str(idx), (x_n[idx],y_n[idx]))

            else:                               # Interior grid points
                if idx <= div * 2:
                    offset = grid_row_pts[row_num-1]
                else:
                    offset = grid_row_pts[row_num]
                K_minus[idx] = K_minus[idx-offset]
                K_plus[idx] = K_plus[idx-1]
                theta_n[idx] = (K_minus[idx] + K_plus[idx])/2
                nu_n[idx] = (K_minus[idx] - K_plus[idx])/2
                M_n[idx] = root_scalar(AT.RS_mach_prandtl_meyer, bracket=[1,10],
                                       args=(nu_n[idx], k)).root
                mu_n[idx] = np.arcsin(1/M_n[idx])
                col[idx] = col[idx-offset]

                m_1 = np.tan((theta_n[idx-1] + mu_n[idx-1] + theta_n[idx] +
                              mu_n[idx])/2)
                x_1 = x_n[idx-1]
                y_1 = y_n[idx-1]
                m_2 = np.tan((theta_n[idx-offset] - mu_n[idx-offset] +
                              theta_n[idx] - mu_n[idx])/2)
                x_2 = x_n[idx-offset]
                y_2 = y_n[idx-offset]
                x_3 = (m_1*x_1 - m_2*x_2 + y_2 - y_1)/(m_1-m_2)
                y_3 = y_1 + m_1*(x_3 - x_1)
                x_n[idx] = x_3
                y_n[idx] = y_3

                if plot_flag:
                    plt.plot([x_n[idx-offset], x_n[idx]],
                             [y_n[idx-offset], y_n[idx]],
                             c=custom_cmap(int(col[idx])), linewidth=1)
                    plt.plot([x_n[idx-1], x_n[idx]], [y_n[idx-1], y_n[idx]],
                             c=custom_cmap(int(col[row_num])), linewidth=1)
                    plt.scatter(x_n[idx], y_n[idx], s=1)
                    plt.annotate(str(idx), (x_n[idx],y_n[idx]))

            idx += 1

    if print_flag:
        for i in range(len(theta_n)):
            print(f" i {i} theta_n {theta_n[i]*180/np.pi:.3f}" +
                  f" nu_n {nu_n[i]*180/np.pi:.3f}" +
                  f" K- {K_minus[i]*180/np.pi:.3f}" +
                  f" K+ {K_plus[i]*180/np.pi:.3f}"+
                  f" M_n {M_n[i]:.3f} mu_n {mu_n[i]*180/np.pi:.3f}" +
                  f" x_n {x_n[i]:.3f} y_n {y_n[i]:.3f} col {col[i]}")

    x_str = np.array(x_str)
    y_str = np.array(y_str)

    # Fit nonlinear regression to MOC contour
    p0 = [1.0, 1.0, 1.0, 1.0, 1.0]
    popt, pcov = curve_fit(high_poly_law, x_str, y_str, p0=p0, maxfev=10000)
    a, b, c, d, e = popt

    # Generate contour using fitted coefficients
    x_str_fit = np.linspace(x_str[0], x_str[-1], res)
    y_str_fit = high_poly_law(x_str_fit, a, b, c, d, e)

    # Fill in gap between expansion and straightening sections
    x_gap = np.linspace(x_expand[-1], x_str[0], int(res/5))
    y_gap = np.linspace(y_expand[-1], y_str[0], int(res/5))

    x_contour = np.concatenate((x_expand,x_gap[1:-1],x_str_fit))
    y_contour = np.concatenate((y_expand,y_gap[1:-1],y_str_fit))

    return x_contour, y_contour, x_n, y_n, M_n

if __name__ == "__main__":
    # Test script
    x_contour, y_contour, x_n, y_n, M_n = gen_MOC_MLN(2.4,.5,1.4,7, True, True)
    plt.figure()
    plt.plot(x_contour, y_contour, 'k-', linewidth=2)

    # Interpolated color mesh based on Mach number
    plt.figure()
    x_n = np.append(x_n, max(x_n))
    y_n = np.append(y_n, 0)
    M_n = np.append(M_n, max(M_n))
    x_i = np.linspace(min(x_n), max(x_n), 300)
    y_i = np.linspace(min(y_n), max(y_n), 300)
    X,Y = np.meshgrid(x_i,y_i)
    Z = griddata((x_n, y_n), M_n, (X, Y), method='cubic')
    c = plt.pcolormesh(X, Y, Z, cmap=plt.cm.RdYlBu)
    plt.scatter(x_n, y_n, c=M_n, edgecolor="k", s=1)  # show original points
    plt.plot(x_contour, y_contour, 'k-', linewidth=2)
    plt.title('Mach Number Contour')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.colorbar(c)
    plt.axis('equal')

    x_contour, y_contour, x_n, y_n, M_n = gen_MOC_FLN(3,.5,1.4,30, True, True)
    plt.figure()
    plt.plot(x_contour, y_contour, 'k-', linewidth=2)
    plt.show()
