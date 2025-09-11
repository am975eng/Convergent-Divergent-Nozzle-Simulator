import sys
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QGridLayout, QLineEdit, QSpacerItem, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
from scipy.optimize import root_scalar
from scipy.optimize import fsolve

class MplCanvas(FigureCanvas):
    """
    Matplotlib canvas generated using Agg engine that can function as a QWidget.
    """
    def __init__(self, parent=None):
        """
        Initializes the instance with a figure and subplot axes and sets style properties.
        """
        self.fig = Figure(figsize=(7, 8))
        self.fig.patch.set_alpha(0)
        self.axes = self.fig.add_subplot(411)
        self.axes_2 = self.fig.add_subplot(412)
        self.axes_3 = self.fig.add_subplot(413)
        self.axes_4 = self.fig.add_subplot(414)
        # self.axes.set_aspect("equal")

        all_axes = [self.axes, self.axes_2, self.axes_3, self.axes_4]
        for ax in all_axes:
            ax.set_facecolor('none')
            ax.spines['top'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.tick_params(colors='white')

        super().__init__(self.fig)
        self.setParent(parent)


class MyWindow(QMainWindow):
    """
    Main application window inherited from QMainWindow.
    """
    def __init__(self):
        super().__init__()
        self.load_styles()
        self.init_UI()

    def load_styles(self):
        """Load styles from QSS file"""
        try:
            # Get the directory where this script is located
            current_dir = Path(__file__).parent
            style_file = current_dir / "style.qss"
            print(f"Loading styles from {style_file}")
           
            with open(style_file, "r") as f:
                stylesheet = f.read()
                self.setStyleSheet(stylesheet)
                
        except FileNotFoundError:
            print("Style file not found. Using default styling.")
        except Exception as e:
            print(f"Error loading styles: {e}")    
    
    def init_UI(self):
        """Sets up UI by creating and positioning widgets."""
        # Set window properties
        self.setWindowTitle("CGT Designer")
        widget = QWidget()
        
        # Create widgets
        DC_label = QLabel("Design Variables")
        prop_label = QLabel("Propellant")
        prop_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.prop_list = QComboBox()
        self.prop_list.addItems(["Air", "CO2", "N2", "Xe"])
        self.prop_list.currentTextChanged.connect(self.update_result)
        pressure_label = QLabel("Chamber Pressure [Pa]")
        pressure_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_chamber = QLineEdit("599844")
        self.P_chamber.textChanged.connect(self.update_result)
        temp_label = QLabel("Chamber Temperature [K]")
        temp_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.T_chamber = QLineEdit("293.15")
        self.T_chamber.textChanged.connect(self.update_result)
        amb_press_label = QLabel("Ambient Pressure [Pa]")
        amb_press_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_amb = QLineEdit("0")
        self.P_amb.textChanged.connect(self.update_result)        
        converg_label = QLabel("Convergence Angle [Deg]")
        converg_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.converg_ang = QLineEdit("45")
        self.converg_ang.textChanged.connect(self.update_result)
        diverg_label = QLabel("Divergence Angle [Deg]")
        diverg_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.diverg_angle = QLineEdit("15")
        self.diverg_angle.textChanged.connect(self.update_result)

        radius_inlet_label = QLabel("Inlet Radius [m]")
        self.radius_inlet = QLineEdit("0.1")
        self.radius_inlet.textChanged.connect(self.update_result)        
        self.radius_throat_label = QLabel("Throat Radius [m]")
        self.radius_throat_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_throat = QLineEdit("0.01")
        self.radius_throat.textChanged.connect(self.update_result)
        self.radius_exit_label = QLabel("Exit Radius [m]")
        self.radius_exit_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_exit = QLineEdit("0.1")
        self.radius_exit.textChanged.connect(self.update_result)
        thrust_design_label = QLabel("Design Thrust [N]")
        self.thrust_design = QLineEdit("0")
        self.thrust_design.textChanged.connect(self.update_result)

        results_label = QLabel("Result Summary")
        self.result_display_label = QLabel("Shock")
        self.result_display_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        P_e_sup_label = QLabel("Supersonic Choked Exit Pressure [Pa]")
        P_e_sup_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_e_sup_val = QLabel(" ")
        self.P_e_sup_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        P_e_sub_label = QLabel("Subsonic Choked Exit Pressure [Pa]")
        P_e_sub_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_e_sub_val = QLabel(" ")
        self.P_e_sub_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        P_e_shock_label = QLabel("Exit Pressure for Exit Shock [Pa]")
        P_e_shock_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_e_shock_val = QLabel(" ")
        self.P_e_shock_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        P_star_shock_label = QLabel("Exit Pressure for Initial Shock [Pa]")
        P_star_shock_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_star_shock_val = QLabel(" ")
        self.P_star_shock_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        m_dot_label = QLabel("Mass Flow Rate [kg/s]")
        m_dot_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.m_dot_val = QLabel(" ")
        self.m_dot_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        thrust_label = QLabel("Actual Thrust [N]")
        thrust_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.thrust_val = QLabel(" ")
        self.thrust_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Create layout and add widgets
        option_layout = QGridLayout()

        option_layout.addWidget(DC_label, 0, 0)
        option_layout.addWidget(prop_label, 1, 0)
        option_layout.addWidget(self.prop_list, 2, 0)
        option_layout.addWidget(pressure_label, 3, 0)
        option_layout.addWidget(self.P_chamber, 4, 0)
        option_layout.addWidget(temp_label, 5, 0)
        option_layout.addWidget(self.T_chamber, 6, 0)
        option_layout.addWidget(converg_label, 7, 0)
        option_layout.addWidget(self.converg_ang, 8, 0)
        option_layout.addWidget(diverg_label, 9, 0)
        option_layout.addWidget(self.diverg_angle, 10, 0)

        option_layout.addWidget(radius_inlet_label, 1, 1)        
        option_layout.addWidget(self.radius_inlet, 2, 1)
        option_layout.addWidget(self.radius_throat_label, 3, 1)
        option_layout.addWidget(self.radius_throat, 4, 1)
        option_layout.addWidget(self.radius_exit_label, 5, 1)
        option_layout.addWidget(self.radius_exit, 6, 1)        
        option_layout.addWidget(amb_press_label, 7, 1)
        option_layout.addWidget(self.P_amb, 8, 1)
        option_layout.addWidget(thrust_design_label, 9, 1)
        option_layout.addWidget(self.thrust_design, 10, 1)
        v_spacer = QSpacerItem(
        20, 40,                       # width, height (the actual numbers don’t matter much)
        QSizePolicy.Policy.Minimum,   # horizontal policy (doesn’t expand sideways)
        QSizePolicy.Policy.Expanding  # vertical policy (absorbs all extra vertical space)
        )
        h_spacer = QSpacerItem(
        40, 20,                       # width, height (the actual numbers don’t matter much)
        QSizePolicy.Policy.Expanding, # horizontal policy (absorbs all extra horizontal space)
        QSizePolicy.Policy.Minimum    # vertical policy (doesn’t expand sideways)
        )
        option_layout.addItem(v_spacer, 11, 0, 1, 2)
        
        option_layout.addWidget(results_label, 12, 0)
        option_layout.addWidget(self.result_display_label, 13, 0, 1, 2)
        option_layout.addWidget(P_e_sup_label, 14, 0)
        option_layout.addWidget(self.P_e_sup_val, 15, 0)
        option_layout.addWidget(P_e_sub_label, 14, 1)
        option_layout.addWidget(self.P_e_sub_val, 15, 1)
        option_layout.addWidget(P_e_shock_label, 16, 0)
        option_layout.addWidget(self.P_e_shock_val, 17, 0)
        option_layout.addWidget(P_star_shock_label, 16, 1)
        option_layout.addWidget(self.P_star_shock_val, 17, 1)
        option_layout.addWidget(m_dot_label, 18, 0)
        option_layout.addWidget(self.m_dot_val, 19, 0)
        option_layout.addWidget(thrust_label, 18, 1)
        option_layout.addWidget(self.thrust_val, 19, 1)

        graphic_layout = QVBoxLayout()
        self.canvas = MplCanvas(self)
        graphic_layout.addWidget(self.canvas)
        
        outer_layout = QHBoxLayout()
        outer_layout.addLayout(option_layout)
        outer_layout.addSpacerItem(h_spacer)
        outer_layout.addLayout(graphic_layout)

        widget.setLayout(outer_layout)
        self.setCentralWidget(widget)

        self.update_result()


    def update_result(self):
        """
        Main function that gets triggered by a UI event. Extracts UI data and recalculates flow thermodynamics.
        """
        def reflect_plot(self):
            lines = self.canvas.axes.get_lines()
            for line in lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()

                self.canvas.axes.plot(x_data, -y_data, 
                    color=line.get_color(), 
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth())  # Slightly transparent

        self.canvas.axes.clear()            
        self.canvas.axes_2.clear()
        self.canvas.axes_3.clear()
        self.canvas.axes_4.clear()
        
        # Gas specific properties       
        if self.prop_list.currentText() == "Air":
            self.R_spec = 287
            self.k=1.4
        elif self.prop_list.currentText() == "CO2":
            self.R_spec = 188.9
            self.k=1.289
        elif self.prop_list.currentText() == "N2":
            self.R_spec = 296.8
            self.k=1.4
        elif self.prop_list.currentText() == "Xe":
            self.R_spec = 63.33
            self.k=1.667

        # Geometry
        r_throat = float(self.radius_throat.text())
        r_inlet = float(self.radius_inlet.text())
        r_outlet = float(self.radius_exit.text())
        A_star = math.pi*(r_throat**2)
        A_inlet = math.pi*(r_inlet**2)
        A_outlet = math.pi*(r_outlet**2)

        converg_angle= np.deg2rad(float(self.converg_ang.text()))
        diverg_angle = np.deg2rad(float(self.diverg_angle.text()))
        converg_length = (r_inlet-r_throat)/np.tan(converg_angle)
        diverg_length = (r_outlet-r_throat)/np.tan(diverg_angle)

        res = 150
        x_conv = -np.flip(np.linspace(0, converg_length, res))
        y_conv = r_throat -x_conv*np.tan(converg_angle)
        x_div = np.linspace(0, diverg_length, res)
        y_div = r_throat + x_div*np.tan(diverg_angle)
        self.P_array = np.zeros(len(x_conv)+len(x_div))
        self.M_array = np.zeros(len(x_conv)+len(x_div))
        self.T_array = np.zeros(len(x_conv)+len(x_div))
            
        # Extract thermo. properties
        T_0 = float(self.T_chamber.text())
        P_0 = float(self.P_chamber.text())
        rho_0 = P_0/(self.R_spec*T_0)
        P_amb = float(self.P_amb.text())

        # Throat Conditions
        T_star = T_0 * (2/(self.k+1))
        P_star = P_0 * ((2/(self.k+1))**(self.k/(self.k-1)))
        self.M_throat = 1

        #self.m_dot = math.pi*(r_throat**2)*P_0*(((self.k/(self.R_spec*T_0)))**0.5)*((2/(self.k+1))**((self.k+1)/(2*(self.k-1))))
               
        def Mach_Press_Isen(M, P_0, P):
            """
            Function to calculate Mach number at a point using isentropic relation and stagnation pressure.
            
            Args:
                M - Mach number to be swept using root_scalar
                p_ratio - Stagnation pressure ratio
            """
            Mach_Press_Isen = (-M)+((((P_0/P)**((self.k-1)/self.k))-1)*(2/(self.k-1)))**0.5
            return Mach_Press_Isen
        
        def Area_Mach_x_y(M_y, M_x, A_y, A_x):
            """
            Isentropic area-Mach relation.
            
            Args:
                M_y - Mach number at y point
                M_x - Mach number at x point
                A_y - Area at y point
                A_x - Area at x point
            """
            expon = (self.k+1)/(self.k-1)
            term_x = 1 + ((self.k-1)/2)*M_x*M_x
            term_y = 1 + ((self.k-1)/2)*M_y*M_y
            Area_Mach_x_y = (-A_y/A_x)+(M_x/M_y)*(((term_y/term_x)**expon)**0.5)
            return Area_Mach_x_y
        
        def calc_isen_press(M,P):
            """
            Uses isentropic relation to calculate pressure at a point using stagnation conditions.
            
            Args:
                M - Mach number at point of interest
                P - Stagnation pressure
            """
            return P / (((1+ ((self.k-1)/2) * M * M))**(self.k/(self.k-1)))

        # Solve for supersonic and subsonic exit conditions
        try:
            M_e_sup = root_scalar(Area_Mach_x_y, bracket=[1,100], args=(self.M_throat,A_outlet,A_star)).root
            M_e_sub = root_scalar(Area_Mach_x_y, bracket=[0.0001,1], args=(self.M_throat,A_outlet,A_star)).root
        except ValueError as e:
            print("Unable to solve for Mach numbers. Expand solver bracket to ensure solution exists.")

        self.P_e_sup = calc_isen_press(M_e_sup,P_0)
        self.P_e_sub = calc_isen_press(M_e_sub,P_0)
        self.P_e_sup_val.setText("{:.4g}".format(self.P_e_sup))
        self.P_e_sub_val.setText("{:.4g}".format(self.P_e_sub))
        
        def iter_div_sect(self):
            """
            Iterates through the divergent section to determine if flow is supersonic or subsonic and iterates shock
            location until back and exit pressure match.
            """
            shock_flag = False  # If shock calc. succeeds iterate with this shock location
            self.x_shock = None # Shock location
            A_x = math.pi*(y_div[-1]**2)
            M_x_sup = root_scalar(Area_Mach_x_y, bracket=[1,100], args=(self.M_throat,A_x,A_star)).root
            P_x = calc_isen_press(M_x_sup,P_0)
            T_x = T_0 * ((1+ ((self.k-1)/2)*M_x_sup*M_x_sup)**-1)
            M_y = ((M_x_sup**2 + (2/(self.k-1)))/((2*self.k/(self.k-1))*M_x_sup*M_x_sup-1))**0.5
            P_y = P_x*(((2*self.k/(self.k+1))*M_x_sup*M_x_sup) - (self.k-1)/(self.k+1))
            P_0_y = calc_isen_press(M_y,P_y) 
            T_0_y = T_0

            A_star_shock = A_x * M_y * (((2/(self.k+1))*(1+((self.k-1)/2)*M_y*M_y))**((-self.k-1)/(2*self.k-2)))
            M_y_e = root_scalar(Area_Mach_x_y, bracket=[0.0001,1], args=(self.M_throat,A_outlet,A_star_shock)).root   
            P_e_shock = calc_isen_press(M_y_e,P_0_y)
            self.P_e_shock_val.setText("{:.1f}".format(P_e_shock))

            A_x = math.pi*(y_div[1]**2)
            M_x_sup = root_scalar(Area_Mach_x_y, bracket=[1,100], args=(self.M_throat,A_x,A_star)).root
            P_x = calc_isen_press(M_x_sup,P_0)
            T_x = T_0 * ((1+ ((self.k-1)/2)*M_x_sup*M_x_sup)**-1)
            M_y = ((M_x_sup**2 + (2/(self.k-1)))/((2*self.k/(self.k-1))*M_x_sup*M_x_sup-1))**0.5
            P_y = P_x*(((2*self.k/(self.k+1))*M_x_sup*M_x_sup) - (self.k-1)/(self.k+1))
            P_0_y = calc_isen_press(M_y,P_y) 
            T_0_y = T_0

            A_star_shock = A_x * M_y * (((2/(self.k+1))*(1+((self.k-1)/2)*M_y*M_y))**((-self.k-1)/(2*self.k-2)))
            M_y_e = root_scalar(Area_Mach_x_y, bracket=[0.0001,1], args=(self.M_throat,A_outlet,A_star_shock)).root   
            P_star_shock = calc_isen_press(M_y_e,P_0_y)
            self.P_star_shock_val.setText("{:.1f}".format(P_star_shock))

            if P_amb >= P_0:
                self.result_display_label.setText("Back pressure high enough to generate reversed flow")
            elif P_amb >= self.P_e_sub:
                # Back pressure is too high for choked subsonic flow
                self.result_display_label.setText("Back pressure too high for choked subsonic flow")
                M_e_sub = root_scalar(Mach_Press_Isen, bracket=[0.0001,1], args=(P_0,P_amb)).root
                self.M_throat = root_scalar(Area_Mach_x_y, bracket=[0.0001,1], args=(M_e_sub,A_star,A_outlet)).root
                
                for index in range(len(x_div)):
                    A_x = math.pi*(y_div[index]**2)
                    shift = index + len(x_conv)

                    M_x = root_scalar(Area_Mach_x_y, bracket=[0.0001,1], args=(M_e_sub, A_x, A_outlet)).root
                    P_x = calc_isen_press(M_x,P_0)
                    T_x = T_0 * ((1+ ((self.k-1)/2)*M_x*M_x)**-1)

                    self.P_array[shift] = P_x
                    self.M_array[shift] = M_x
                    self.T_array[shift] = T_x

            elif P_amb > P_e_shock:
                # Back pressure is low enough for choked supersonic flow with possible normal shock
                for index in range(len(x_div)):
                    A_x = math.pi*(y_div[index]**2)
                    shift = index + len(x_conv)
                    if not shock_flag:
                        # Pre shock wave prop
                        M_x_sup = root_scalar(Area_Mach_x_y, bracket=[1,100], args=(self.M_throat,A_x,A_star)).root
                        P_x = calc_isen_press(M_x_sup,P_0)
                        T_x = T_0 * ((1+ ((self.k-1)/2)*M_x_sup*M_x_sup)**-1)
                        self.P_array[shift] = P_x
                        self.M_array[shift] = M_x_sup
                        self.T_array[shift] = T_x

                        # Post shock wave prop
                        M_y = ((M_x_sup**2 + (2/(self.k-1)))/((2*self.k/(self.k-1))*M_x_sup*M_x_sup-1))**0.5
                        P_y = P_x*(((2*self.k/(self.k+1))*M_x_sup*M_x_sup) - (self.k-1)/(self.k+1))
                        P_0_y = calc_isen_press(M_y,P_y) 
                        T_0_y = T_0

                        A_star_shock = A_x * M_y * (((2/(self.k+1))*(1+((self.k-1)/2)*M_y*M_y))**((-self.k-1)/(2*self.k-2)))
                        M_y_e = root_scalar(Area_Mach_x_y, bracket=[0.0001,1], args=(self.M_throat,A_outlet,A_star_shock)).root   
                        P_y_e = calc_isen_press(M_y_e,P_0_y)
                        T_y_e = T_0_y * ((1+ ((self.k-1)/2)*M_y_e*M_y_e)**-1)

                        if abs((P_y_e - P_amb)/P_y_e) < 0.1:
                            print('Shock location calculated')
                            shock_flag = True
                            self.x_shock = x_div[index]
                            self.result_display_label.setText(f'Normal shock generated in divergent section at {self.x_shock:.3f} m')
                    elif shock_flag:
                        M_y_curr = root_scalar(Area_Mach_x_y, bracket=[0.0001,1], args=(self.M_throat,A_x,A_star_shock)).root
                        P_y_curr = calc_isen_press(M_y_curr, P_0_y)
                        T_y_curr = T_0_y * ((1+ ((self.k-1)/2)*M_y_curr*M_y_curr)**-1)
                        self.P_array[shift] = P_y_curr
                        self.M_array[shift] = M_y_curr
                        self.T_array[shift] = T_y_curr
                
                
                if abs((self.P_e_sup - P_amb)/self.P_e_sup)<0.5 or (self.P_e_sup<500 and P_amb<50):
                    self.result_display_label.setText('No normal shocks generated in the nozzle')
            elif P_amb > self.P_e_sup:
                # Overexpanded nozzle with shockwaves in exhaust
                self.result_display_label.setText('Overexpanded exhaust with shockwaves in exhaust')    
                for index in range(len(x_div)):
                    A_x = math.pi*(y_div[index]**2)
                    shift = index + len(x_conv)
                    M_x_sup = root_scalar(Area_Mach_x_y, bracket=[1,100], args=(self.M_throat,A_x,A_star)).root
                    P_x = calc_isen_press(M_x_sup,P_0)
                    T_x = T_0 * ((1+ ((self.k-1)/2)*M_x_sup*M_x_sup)**-1)
                    self.P_array[shift] = P_x
                    self.M_array[shift] = M_x_sup
                    self.T_array[shift] = T_x

                # Oblique shock
                def over_shock_eqn(vars, M_1, P_1, P_2):
                    beta, theta = vars
                    M_1_n = M_1 * np.sin(beta)
                    eqn1 = -(P_2/P_1) + (2*self.k*M_1_n*M_1_n-self.k+1)/(self.k+1)
                    eqn2 = -np.tan(theta) + (2*(1/np.tan(beta))*(M_1*M_1*np.sin(beta)*np.sin(beta)-1))/(M_1*M_1*(self.k+np.cos(2*beta))+2)
                    #M_2_n = M_2 * np.sin(beta-theta)
                    return [eqn1, eqn2]
                
                beta, theta = fsolve(over_shock_eqn,[np.radians(45),np.radians(45)],args=(self.M_array[-1],self.P_array[-1],P_amb))
                # print(f"Beta: {beta*180/np.pi:.2f} degrees, Theta: {theta*180/np.pi:.2f} degrees")
                x_over_shock_1 = x_div[-1]
                y_over_shock_1 = y_div[-1]
                x_over_shock_2 = x_over_shock_1 + y_over_shock_1 * np.cos(beta)
                y_over_shock_2 = y_over_shock_1 - y_over_shock_1 * np.sin(beta)
                self.canvas.axes.plot([x_over_shock_1,x_over_shock_2],[y_over_shock_1,y_over_shock_2], 'g-')
                
            else:
                # Underexpanded nozzle
                self.result_display_label.setText('Underexpanded supersonic exhaust')
                for index in range(len(x_div)):
                    A_x = math.pi*(y_div[index]**2)
                    shift = index + len(x_conv)
                    M_x_sup = root_scalar(Area_Mach_x_y, bracket=[1,100], args=(self.M_throat,A_x,A_star)).root
                    P_x = calc_isen_press(M_x_sup,P_0)
                    T_x = T_0 * ((1+ ((self.k-1)/2)*M_x_sup*M_x_sup)**-1)
                    self.P_array[shift] = P_x
                    self.M_array[shift] = M_x_sup
                    self.T_array[shift] = T_x


            self.P_e=self.P_array[-1]
            self.M_e=self.M_array[-1]
            self.T_e=self.T_array[-1]
            self.rho_e = self.P_e/(self.R_spec*self.T_e)
            self.c_e = (self.k*self.R_spec*self.T_e)**0.5
            self.m_dot = self.rho_e * A_outlet * self.M_e * self.c_e 
            self.m_dot_val.setText(f"{self.m_dot:.3f} kg/s")

        iter_div_sect(self)
        for index in range(len(x_conv)):
            A_x = math.pi*(y_conv[index]**2)
            M_x_sub = root_scalar(Area_Mach_x_y, bracket=[0.0001,1], args=(self.M_throat,A_x,A_star)).root
            P_x = calc_isen_press(M_x_sub,P_0)
            T_x = T_0 * ((1+ ((self.k-1)/2)*M_x_sub*M_x_sub)**-1)

            self.P_array[index] = P_x
            self.M_array[index] = M_x_sub
            self.T_array[index] = T_x
             
        # Thrust Calc
        c_e = math.sqrt(self.k*self.R_spec*self.T_e)
        V_e = self.M_e * c_e
        Thr = self.m_dot * V_e + (self.P_e - P_amb) * A_outlet
        self.thrust_val.setText("{:.3g}".format(Thr))
        
        # Plot data
        self.canvas.axes.plot(x_conv, y_conv, 'b-', linewidth=2)
        self.canvas.axes.plot(x_div, y_div, 'b-', linewidth=2)   
        
        reflect_plot(self)
        self.canvas.axes.grid(True)
        self.canvas.axes.title.set_color('white')
        

        def gen_cmap_plot(x,y,ax):
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap = plt.cm.RdYlBu 
            norm = mpl.colors.Normalize(vmin=np.min(-y), vmax=np.max(-y))  # <-- added
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
            lc.set_array(-y) 
            
            ax.add_collection(lc)
            range = np.max(y) - np.min(y)
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(np.min(y) - 0.1*range, np.max(y)+0.15*range)
 
        gen_cmap_plot(np.concatenate([x_conv, x_div]), self.P_array, self.canvas.axes_2)
        gen_cmap_plot(np.concatenate([x_conv, x_div]), self.M_array, self.canvas.axes_3)
        gen_cmap_plot(np.concatenate([x_conv, x_div]), self.T_array, self.canvas.axes_4)

        self.canvas.axes.set_ylabel('Y Position [m]', color='white')
        self.canvas.axes_2.set_ylabel('Pressure [Pa]', color='white')
        self.canvas.axes_3.set_ylabel('Mach Number', color='white')
        self.canvas.axes_4.set_ylabel('Temperature [K]', color='white')
        self.canvas.axes_4.set_xlabel('X Position [m]', color='white')
        # Refresh canvas
        self.canvas.draw()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())