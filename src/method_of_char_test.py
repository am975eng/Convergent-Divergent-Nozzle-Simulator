from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

import thermodynamics.aero_thermo as AT
import thermodynamics.method_of_char as MOC

x_contour, y_contour, x_n, y_n, M_n = MOC.gen_MOC_MLN(
    2.4, 0.5, 1.4, 7, True, True
)

# Interpolated color mesh based on Mach number
plt.figure()
x_n = np.append(x_n, max(x_n))
y_n = np.append(y_n, 0)
M_n = np.append(M_n, max(M_n))
x_i = np.linspace(min(x_n), max(x_n), 300)
y_i = np.linspace(min(y_n), max(y_n), 300)
X, Y = np.meshgrid(x_i, y_i)
Z = griddata((x_n, y_n), M_n, (X, Y), method="cubic")
c = plt.pcolormesh(X, Y, Z, cmap=plt.cm.RdYlBu)
plt.scatter(x_n, y_n, c=M_n, edgecolor="k", s=1)  # show original points
plt.plot(x_contour, y_contour, "k-", linewidth=2)
plt.title("Mach Number Contour")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.colorbar(c)
plt.axis("equal")

x_contour, y_contour, x_n, y_n, M_n = MOC.gen_MOC_FLN(
    2.5, 0.5, 1.4, 12, True, True
)

x_contour, y_contour = MOC.gen_rao_bell(40, 40 * (25**0.5))
plt.figure()
plt.plot(x_contour, y_contour, "k-", linewidth=2)
plt.title("Rao Bell Nozzle")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.show()
