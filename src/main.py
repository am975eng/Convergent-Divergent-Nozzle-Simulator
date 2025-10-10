import sys
from pathlib import Path
import math

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSpacerItem,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QPushButton)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import numpy as np
from scipy.optimize import (
    fsolve,
    root_scalar)
from CoolProp.CoolProp import PropsSI, PhaseSI, AbstractState
import CoolProp.CoolProp as CP

import Aero_Thermo as AT
import Method_Of_Char as MOC
from Optimizer import ADAM_Optimizer


class MplCanvas(FigureCanvas):
    """
    Matplotlib canvas generated using Agg engine that can function as a
    QWidget.
    """
    def __init__(self, parent=None):
        """
        Initializes the instance with a figure and subplot axes and sets 
        style properties.
        """
        self.fig = Figure(figsize=(9, 9),constrained_layout=True)
        self.fig.patch.set_alpha(0)
        self.axes = self.fig.add_subplot(321)
        self.axes_press = self.fig.add_subplot(322)
        self.axes_mass = self.fig.add_subplot(323)
        self.axes_mach = self.fig.add_subplot(324)
        self.axes_depress = self.fig.add_subplot(325)
        self.axes_temp = self.fig.add_subplot(326)

        self.axes.grid(True)
        self.axes.title.set_color('white')

        all_axes = [self.axes, self.axes_press, self.axes_mass, self.axes_mach, self.axes_depress, self.axes_temp]
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

        self._debounce = QTimer(singleShot=True, interval=400)
        self._debounce.timeout.connect(self.update_result)

    def _schedule_update(self):
        # Start the debounce timer to prevent rapid updates
        self._debounce.start()

    def load_styles(self):
        """Load styles from QSS file"""
        try:
            # Get the directory where this script is located
            current_dir = Path(__file__).parent
            style_file = current_dir / "style.qss"

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
        self.prop_list.currentTextChanged.connect(self._schedule_update)
        noz_label = QLabel("Nozzle Geometry Type")
        noz_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.noz_type_list = QComboBox()
        self.noz_type_list.addItems(["MOC Full Length Nozzle",
                                     "MOC Minimum Length Nozzle", "Conical"])
        self.noz_type_list.currentTextChanged.connect(self._schedule_update)
        P_chamber_label = QLabel("Chamber Pressure [Pa]")
        P_chamber_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_chamber_val = QLineEdit("5000")
        self.P_chamber_val.textChanged.connect(self._schedule_update)
        T_chamber_label = QLabel("Chamber Temperature [K]")
        T_chamber_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.T_chamber_val = QLineEdit("293.15")
        self.T_chamber_val.textChanged.connect(self._schedule_update)
        P_amb_label = QLabel("Ambient Pressure [Pa]")
        P_amb_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_amb_val = QLineEdit("0")
        self.P_amb_val.textChanged.connect(self._schedule_update)
        converg_label = QLabel("Convergence Angle [Deg]")
        converg_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.converg_ang_val = QLineEdit("45")
        self.converg_ang_val.textChanged.connect(self._schedule_update)
        diverg_label = QLabel("Divergence Angle [Deg]")
        diverg_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.diverg_angle_val = QLineEdit("15")
        self.diverg_angle_val.textChanged.connect(self._schedule_update)
        self.optimize_button = QPushButton("Optimize Geometry")
        self.optimize_button.clicked.connect(self.calc_opt_geom)

        length_inlet_label = QLabel("Inlet Length [m]")
        length_inlet_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.length_inlet_val = QLineEdit("0.1")
        self.length_inlet_val.textChanged.connect(self._schedule_update)
        radius_inlet_label = QLabel("Inlet Radius [m]")
        radius_inlet_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_inlet_val = QLineEdit("0.1")
        self.radius_inlet_val.textChanged.connect(self._schedule_update)
        self.radius_throat_label = QLabel("Throat Radius [m]")
        self.radius_throat_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_throat_val = QLineEdit("0.08")
        self.radius_throat_val.textChanged.connect(self._schedule_update)
        self.radius_exit_label = QLabel("Exit Radius [m]")
        self.radius_exit_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_exit_val = QLineEdit("0.1")
        self.radius_exit_val.textChanged.connect(self._schedule_update)
        M_exit_label = QLabel("Exit Mach Number")
        M_exit_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.M_exit_val = QLineEdit("2.0")
        self.M_exit_val.textChanged.connect(self._schedule_update)
        thrust_design_label = QLabel("Design Thrust [N]")
        thrust_design_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.thrust_design_val = QLineEdit("100")
        self.depress_button = QPushButton("Depressurize")
        self.depress_button.clicked.connect(self.calc_depress)

        # Results Widgets
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
        P_exit_label = QLabel("Exit Pressure [Pa]")
        P_exit_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_exit_val = QLabel(" ")
        self.P_exit_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        ISP_label = QLabel("Specific Impulse [s]")
        ISP_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.ISP_val = QLabel(" ")
        self.ISP_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        thrust_label = QLabel("Actual Thrust [N]")
        thrust_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.thrust_val = QLabel(" ")
        self.thrust_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Create layout and add widgets
        option_layout = QGridLayout()
        add_to_layout = lambda col, start, *widgets: [
            option_layout.addWidget(
                widget, i+start, col) for i, widget in enumerate(widgets)]
        add_to_layout(0, 0, DC_label, prop_label, self.prop_list, noz_label,
                      self.noz_type_list, P_chamber_label, self.P_chamber_val,
                      T_chamber_label, self.T_chamber_val, converg_label,
                      self.converg_ang_val, diverg_label,
                      self.diverg_angle_val, self.optimize_button)
        add_to_layout(1, 1, length_inlet_label, self.length_inlet_val, 
                      radius_inlet_label, self.radius_inlet_val, 
                      self.radius_throat_label, self.radius_throat_val,
                      self.radius_exit_label, self.radius_exit_val,
                      M_exit_label, self.M_exit_val, P_amb_label,
                      self.P_amb_val, thrust_design_label,
                      self.thrust_design_val, self.depress_button)
        v_spacer = QSpacerItem(
        20, 40,
        QSizePolicy.Policy.Minimum,   # Doesn’t expand sideways
        QSizePolicy.Policy.Expanding  # Absorbs all extra vertical space
        )
        h_spacer = QSpacerItem(
        40, 20,
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Minimum
        )
        option_layout.addItem(v_spacer, 16, 0, 1, 2)

        option_layout.addWidget(results_label, 17, 0)
        option_layout.addWidget(self.result_display_label, 18, 0, 1, 2)
        add_to_layout(0,19, P_e_sup_label, self.P_e_sup_val, P_e_shock_label,
                      self.P_e_shock_val, m_dot_label, self.m_dot_val,
                      ISP_label, self.ISP_val)
        add_to_layout(1,19, P_e_sub_label, self.P_e_sub_val,
                      P_star_shock_label, self.P_star_shock_val, P_exit_label,
                      self.P_exit_val, thrust_label, self.thrust_val)

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

    def extract_UI_data(self):
        self.fluid = self.prop_list.currentText()
        if self.fluid == "Air":
            self.R_spec = 287
            self.k=1.4
        elif self.fluid == "CO2":
            self.R_spec = 188.9
            self.k=1.289
        elif self.fluid == "N2":
            self.R_spec = 296.8
            self.k=1.4
        elif self.fluid == "Xe":
            self.R_spec = 63.33
            self.k=1.667

        # Extract thermo. properties
        self.T_0 = float(self.T_chamber_val.text())
        self.P_0 = float(self.P_chamber_val.text())
        rho_0 = self.P_0/(self.R_spec*self.T_0)
        self.P_amb = float(self.P_amb_val.text())

        # Throat Conditions
        T_star = self.T_0 * (2/(self.k+1))
        P_star = self.P_0 * ((2/(self.k+1))**(self.k/(self.k-1)))
        self.M_throat = 1

        # Geometry
        self.r_throat = float(self.radius_throat_val.text())
        self.r_inlet = float(self.radius_inlet_val.text())
        self.r_outlet = float(self.radius_exit_val.text())
        self.converg_angle= np.deg2rad(float(self.converg_ang_val.text()))
        self.diverg_angle = np.deg2rad(float(self.diverg_angle_val.text()))
        self.len_inlet = float(self.length_inlet_val.text())

    def calc_thermo(self):
        self.A_star = math.pi*(self.r_throat**2)
        self.A_inlet = math.pi*(self.r_inlet**2)
        self.A_outlet = math.pi*(self.r_outlet**2)
        res = 150
        converg_length = (self.r_inlet-self.r_throat)/np.tan(
            self.converg_angle)
        diverg_length = (self.r_outlet-self.r_throat)/np.tan(
            self.diverg_angle)
        self.x_conv = -np.flip(np.linspace(0, converg_length, res))
        self.y_conv = self.r_throat -self.x_conv*np.tan(self.converg_angle)
        
        if self.noz_type_list.currentIndex() == 0:
            M_exit = float(self.M_exit_val.text())
            self.x_div, self.y_div, *_ = MOC.gen_MOC_FLN(M_exit, self.r_throat,
                                                         k=self.k, div=50)
            self.r_outlet = np.max(self.y_div)
            self.A_outlet = math.pi*(self.r_outlet**2)
            self.converg_ang_val.setReadOnly(True)
            self.diverg_angle_val.setReadOnly(True)
            self.radius_exit_val.setReadOnly(True)
            self.M_exit_val.setReadOnly(False)
        elif self.noz_type_list.currentIndex() == 1:
            M_exit = float(self.M_exit_val.text())
            self.x_div, self.y_div, *_ = MOC.gen_MOC_MLN(M_exit, self.r_throat,
                                                         k=self.k, div=50)
            self.r_outlet = np.max(self.y_div)
            self.A_outlet = math.pi*(self.r_outlet**2)
            self.converg_ang_val.setReadOnly(True)
            self.diverg_angle_val.setReadOnly(True)
            self.radius_exit_val.setReadOnly(True)
            self.M_exit_val.setReadOnly(False)
        elif self.noz_type_list.currentIndex() == 2:
            self.converg_ang_val.setReadOnly(False)
            self.diverg_angle_val.setReadOnly(False)
            self.radius_exit_val.setReadOnly(False)
            self.M_exit_val.setReadOnly(True)
            self.x_div = np.linspace(0, diverg_length, res)
            self.y_div = self.r_throat + self.x_div*np.tan(self.diverg_angle)

        grey_out_style = """QLineEdit[readOnly="true"]
            {background-color: #a3a3a3; color: white;}
            QLineEdit[readOnly="false"]
            { background-color: white; color: black;}"""
        self.converg_ang_val.setStyleSheet(grey_out_style)
        self.diverg_angle_val.setStyleSheet(grey_out_style)
        self.radius_exit_val.setStyleSheet(grey_out_style)
        self.M_exit_val.setStyleSheet(grey_out_style)

        try:
            M_e_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                  args=(self.M_throat, self.A_outlet,
                                        self.A_star, self.k)).root
            M_e_sub = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                  args=(self.M_throat, self.A_outlet,
                                        self.A_star, self.k)).root
        except ValueError as e:
            print("Unable to solve for Mach numbers." +
                  "Expand solver bracket to ensure solution exists.")

        # Preallocation
        self.P_array = np.zeros(len(self.x_conv)+len(self.x_div))
        self.M_array = np.zeros(len(self.x_conv)+len(self.x_div))
        self.T_array = np.zeros(len(self.x_conv)+len(self.x_div))

        self.P_e_sup = AT.calc_isen_press(M_e_sup,self.P_0,self.k)
        self.P_e_sub = AT.calc_isen_press(M_e_sub,self.P_0,self.k)
        self.P_e_sup_val.setText("{:.4g}".format(self.P_e_sup))
        self.P_e_sub_val.setText("{:.4g}".format(self.P_e_sub))

        def iter_div_sect():
            """
            Iterates through the divergent section to determine if flow is
            supersonic or subsonic and iterates shock location until back and
            exit pressure match.
            """
            shock_flag = False  # Checks if shock was calculated
            self.x_shock = None # Shock location

            # Solve for shock at exit of nozzle
            A_x = math.pi*(self.y_div[-1]**2)
            # Mach number before shock
            M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                  args=(self.M_throat, A_x, self.A_star,
                                        self.k)).root
            P_x = AT.calc_isen_press(M_x_sup,self.P_0,self.k)
            T_x = AT.calc_isen_temp(M_x_sup,self.T_0,self.k)
            M_y,P_y = AT.calc_M_P_normal(M_x_sup,P_x,self.k)
            self.P_0_y = AT.calc_isen_stag_press(M_y,P_y,self.k)
            self.T_0_y = self.T_0

            # New critical area using post shock conditions
            self.A_star_shock = A_x * M_y * (
                ((2/(self.k+1))*(1+((self.k-1)/2)*M_y*M_y))**(
                    (-self.k-1)/(2*self.k-2)))
            M_y_e = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.00001,1],
                                args=(self.M_throat,self.A_outlet,
                                      self.A_star_shock,self.k)).root
            P_e_shock = AT.calc_isen_press(M_y_e,self.P_0_y,self.k)
            self.P_e_shock_val.setText("{:.1f}".format(P_e_shock))

            # Solve for shock at onset of divergent section
            A_x = math.pi*(self.y_div[1]**2)
            M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                  args=(self.M_throat, A_x, self.A_star,
                                        self.k)).root
            P_x = AT.calc_isen_press(M_x_sup,self.P_0,self.k)
            T_x = AT.calc_isen_temp(M_x_sup,self.T_0,self.k)
            M_y,P_y = AT.calc_M_P_normal(M_x_sup,P_x,self.k)
            self.P_0_y = AT.calc_isen_stag_press(M_y,P_y,self.k)
            self.T_0_y = self.T_0

            self.A_star_shock = A_x * M_y * (
                ((2/(self.k+1))*(1+((self.k-1)/2)*M_y*M_y))**(
                    (-self.k-1)/(2*self.k-2)))
            M_y_e = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                args=(self.M_throat,self.A_outlet,
                                      self.A_star_shock,self.k)).root
            P_star_shock = AT.calc_isen_press(M_y_e,self.P_0_y,self.k)
            self.P_star_shock_val.setText("{:.1f}".format(P_star_shock))

            if abs(self.P_amb-self.P_e_sup)/self.P_e_sup < 0.1 or (
                self.P_amb < 50 and self.P_e_sup < 100):
                # Check if back pressure and supersonic exit pressure are close
                # enough for perfect expansion
                self.result_display_label.setText(
                    'Perfectly expanded supersonic exhaust!')
                for index in range(len(self.x_div)):
                    A_x = math.pi*(self.y_div[index]**2)
                    shift = index + len(self.x_conv)
                    M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                          args=(self.M_throat,A_x,
                                                self.A_star,self.k)).root
                    P_x = AT.calc_isen_press(M_x_sup,self.P_0,self.k)
                    T_x = AT.calc_isen_temp(M_x_sup,self.T_0,self.k)
                    self.P_array[shift] = P_x
                    self.M_array[shift] = M_x_sup
                    self.T_array[shift] = T_x

            elif self.P_amb >= self.P_0:
                self.result_display_label.setText(
                    "Back pressure high enough to generate reversed flow")
            elif self.P_amb >= self.P_e_sub:
                # Back pressure is too high for choked subsonic flow
                self.result_display_label.setText(
                    "Back pressure too high for choked subsonic flow")
                M_e_sub = root_scalar(
                    AT.RS_Mach_Press_Isen,
                    bracket=[0.0001,1],
                    args=(0,self.P_amb,self.P_0,self.k)).root
                self.M_throat = root_scalar(AT.RS_Area_Mach_X_Y,
                                            bracket=[0.0001,1], 
                                            args=(M_e_sub, self.A_star,
                                                  self.A_outlet, self.k)).root
                
                for index in range(len(self.x_div)):
                    A_x = math.pi*(self.y_div[index]**2)
                    shift = index + len(self.x_conv)

                    M_x = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                      args=(M_e_sub, A_x, self.A_outlet, 
                                            self.k)).root
                    P_x = AT.calc_isen_press(M_x,self.P_0,self.k)
                    T_x = AT.calc_isen_temp(M_x,self.T_0,self.k)

                    self.P_array[shift] = P_x
                    self.M_array[shift] = M_x
                    self.T_array[shift] = T_x

            elif self.P_amb > P_e_shock:
                # Back pressure is low enough for choked supersonic flow with
                # possible normal shock
                for index in range(len(self.x_div)):
                    A_x = math.pi*(self.y_div[index]**2)
                    shift = index + len(self.x_conv)
                    if not shock_flag:
                        # Pre shock wave prop
                        M_x_sup = root_scalar(
                            AT.RS_Area_Mach_X_Y,
                            bracket=[1,100],
                            args=(self.M_throat, A_x, self.A_star, self.k)
                            ).root
                        P_x = AT.calc_isen_press(M_x_sup,self.P_0,self.k)
                        T_x = AT.calc_isen_temp(M_x_sup,self.T_0,self.k)
                        self.P_array[shift] = P_x
                        self.M_array[shift] = M_x_sup
                        self.T_array[shift] = T_x

                        # Post shock wave prop
                        M_y, P_y = AT.calc_M_P_normal(M_x_sup,P_x,self.k)
                        self.P_0_y = AT.calc_isen_stag_press(M_y,P_y,self.k)
                        self.T_0_y = self.T_0

                        self.A_star_shock = A_x * M_y * (
                            ((2/(self.k+1))*(1+((self.k-1)/2)*M_y*M_y))**(
                                (-self.k-1)/(2*self.k-2)))
                        M_y_e = root_scalar(
                            AT.RS_Area_Mach_X_Y,
                            bracket=[0.0001,1], 
                            args=(self.M_throat, self.A_outlet,
                                  self.A_star_shock, self.k)).root   
                        P_y_e = AT.calc_isen_press(M_y_e, self.P_0_y, self.k)
                        T_y_e = self.T_0_y * ((1+((self.k-1)/2)*M_y_e*M_y_e)
                                              **-1)

                        if abs((P_y_e - self.P_amb)/P_y_e) < 0.1:
                            print('Shock location calculated')
                            shock_flag = True
                            self.x_shock = self.x_div[index]
                            self.result_display_label.setText(
                                'Normal shock generated in divergent section' +
                                f'at {self.x_shock:.3f} m')
                    elif shock_flag:
                        M_y_curr = root_scalar(
                            AT.RS_Area_Mach_X_Y,
                            bracket=[0.0001,1],
                            args=(self.M_throat, A_x, self.A_star_shock, 
                                  self.k)).root
                        P_y_curr = AT.calc_isen_press(M_y_curr, self.P_0_y, 
                                                      self.k)
                        T_y_curr = AT.calc_isen_temp(M_y_curr, self.T_0_y, 
                                                     self.k)
                        self.P_array[shift] = P_y_curr
                        self.M_array[shift] = M_y_curr
                        self.T_array[shift] = T_y_curr

            elif self.P_amb > self.P_e_sup:
                # Overexpanded nozzle with shockwaves in exhaust
                self.result_display_label.setText(
                    'Overexpanded exhaust with shockwaves in exhaust')
                for index in range(len(self.x_div)):
                    A_x = math.pi*(self.y_div[index]**2)
                    shift = index + len(self.x_conv)
                    M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                          args=(self.M_throat, A_x,
                                                self.A_star, self.k)).root
                    P_x = AT.calc_isen_press(M_x_sup,self.P_0,self.k)
                    T_x = AT.calc_isen_temp(M_x_sup,self.T_0,self.k)
                    self.P_array[shift] = P_x
                    self.M_array[shift] = M_x_sup
                    self.T_array[shift] = T_x

                beta, theta = fsolve(
                    AT.FS_oblique_angle,
                    [np.radians(45),np.radians(45)],
                    args=(self.M_array[-1], self.P_array[-1], self.P_amb, 
                          self.k))
                x_over_shock_1 = self.x_div[-1]
                y_over_shock_1 = self.y_div[-1]
                hyp = y_over_shock_1 * 0.6
                x_over_shock_2 = x_over_shock_1 + hyp * np.cos(beta)
                y_over_shock_2 = y_over_shock_1 - hyp * np.sin(beta)
                self.canvas.axes.plot([x_over_shock_1,x_over_shock_2],
                                      [y_over_shock_1,y_over_shock_2], 'r-')

            else:
                # Underexpanded nozzle
                self.result_display_label.setText(
                    'Underexpanded supersonic exhaust')
                
                # Divergent section - Supersonic flow
                for index in range(len(self.x_div)):
                    A_x = math.pi*(self.y_div[index]*self.y_div[index])
                    shift = index + len(self.x_conv)
                    M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                          args=(self.M_throat, A_x,
                                                self.A_star, self.k)).root
                    P_x = AT.calc_isen_press(M_x_sup,self.P_0,self.k)
                    T_x = AT.calc_isen_temp(M_x_sup,self.T_0,self.k)
                    self.P_array[shift] = P_x
                    self.M_array[shift] = M_x_sup
                    self.T_array[shift] = T_x


                # Prandtl-Meyer Expansion Fan
                nu_1 = AT.calc_prandtl_meyer(self.M_array[-1],self.k)
                M_2 = root_scalar(AT.RS_Mach_Press_Isen, bracket=[1,100],
                                  args=(self.M_array[-1],self.P_amb+.1,
                                        self.P_array[-1],self.k)).root
                nu_2 = AT.calc_prandtl_meyer(M_2,self.k)
                mu_1 = np.arcsin(1/self.M_array[-1])
                mu_2 = np.arcsin(1/M_2)
                theta = nu_2 - nu_1

                # Plotting fan by sweeping from mu_1 to mu_2
                x_under_shock_1 = self.x_div[-1]
                y_under_shock_1 = self.y_div[-1]
                hyp = y_under_shock_1 * .6
                theta_nu_2 = theta - mu_2
                theta_range = np.linspace(mu_1, theta_nu_2,3)
                for i in range(len(theta_range)):
                    x_under_shock_2 = x_under_shock_1 + np.cos(
                        theta_range[i]) * hyp
                    y_under_shock_2 = y_under_shock_1 - np.sin(
                        theta_range[i]) * hyp
                    self.canvas.axes.plot([x_under_shock_1,x_under_shock_2],
                                          [y_under_shock_1,y_under_shock_2],
                                          'g-')

            # Exit area calculations
            self.P_e=self.P_array[-1]
            self.M_e=self.M_array[-1]
            self.T_e=self.T_array[-1]
            self.rho_e = self.P_e/(self.R_spec*self.T_e)
            c_e = (self.k*self.R_spec*self.T_e)**0.5
            V_e = self.M_e * c_e
            self.m_dot = self.rho_e * self.A_outlet * self.M_e * c_e
            self.thr = self.m_dot * V_e + (
                self.P_e - self.P_amb) * self.A_outlet
            self.ISP = self.thr/(self.m_dot*9.81)

            self.m_dot_val.setText(f"{self.m_dot:.3g}")
            self.P_exit_val.setText(f"{self.P_e:.3g}")
            self.ISP_val.setText(f"{self.ISP:.3g}")
            self.thrust_val.setText(f"{self.thr:.3g}")

        iter_div_sect()

        # Convergent section
        for index in range(len(self.x_conv)):
            A_x = math.pi*(self.y_conv[index]**2)
            M_x_sub = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                  args=(self.M_throat, A_x, self.A_star,
                                        self.k)).root
            P_x = AT.calc_isen_press(M_x_sub,self.P_0,self.k)
            T_x = self.T_0 * ((1+ ((self.k-1)/2)*M_x_sub*M_x_sub)**-1)

            self.P_array[index] = P_x
            self.M_array[index] = M_x_sub
            self.T_array[index] = T_x

    def plot_data(self):
        def reflect_plot():
            """
            Reflects the plot on the opposite side of the x-axis
            """
            lines = self.canvas.axes.get_lines()
            for line in lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()

                self.canvas.axes.plot(x_data, -y_data,
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth())

        self.canvas.axes.plot([self.x_conv[0], self.x_conv[0]-self.len_inlet],
                              [self.y_conv[0], self.y_conv[0]],
                              'b-', linewidth=2)
        self.canvas.axes.plot(
            [self.x_conv[0]-self.len_inlet, self.x_conv[0]-self.len_inlet],
            [self.y_conv[0], 0], 'b-', linewidth=2)
        self.canvas.axes.plot(self.x_conv, self.y_conv, 'b-', linewidth=2)
        self.canvas.axes.plot(self.x_div, self.y_div, 'b-', linewidth=2)
        reflect_plot()

        def gen_cmap_plot(x,y,ax):
            """
            Generates a colormap plot of the given data.

            Inputs:
                x (numpy array) - X-axis data
                y (numpy array) - Y-axis data
                ax (matplotlib axis) - Axis to plot on
            """
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap = plt.cm.RdYlBu 
            norm = mpl_colors.Normalize(vmin=np.min(-y), vmax=np.max(-y))
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
            lc.set_array(-y) 
            
            ax.add_collection(lc)
            span = np.max(y) - np.min(y)
            ax.set_xlim(np.min(x), np.max(x))
            ax.set_ylim(np.min(y) - 0.1*span, np.max(y)+0.15*span)
 
        gen_cmap_plot(np.concatenate([self.x_conv, self.x_div]), self.P_array,
                      self.canvas.axes_press)
        gen_cmap_plot(np.concatenate([self.x_conv, self.x_div]), self.M_array,
                      self.canvas.axes_mach)
        gen_cmap_plot(np.concatenate([self.x_conv, self.x_div]), self.T_array,
                      self.canvas.axes_temp)

        self.canvas.axes.set_ylabel('Y Position [m]', color='white')
        self.canvas.axes_press.set_ylabel('Pressure [Pa]', color='white')
        self.canvas.axes_mach.set_ylabel('Mach Number', color='white')
        self.canvas.axes_temp.set_ylabel('Temperature [K]', color='white')
        self.canvas.axes_temp.set_xlabel('X Position [m]', color='white')

        # Refresh canvas
        self.canvas.draw()

    def update_result(self):
        """
        Main function that gets triggered by a UI event. Extracts UI data and
        recalculates flow thermodynamics.
        """
        self.canvas.axes.clear()
        self.canvas.axes_press.clear()
        self.canvas.axes_mach.clear()
        self.canvas.axes_temp.clear()

        # Extract UI data
        self.extract_UI_data()

        # Thermodynamics
        self.calc_thermo()     

        # Load styles
        #self.load_styles()

        # Plot data
        self.plot_data()

    def calc_depress(self):
        m_dot_curr = self.m_dot
        time_step = .0001
        t = 0

        # Initial Conditions
        P_0_init = self.P_0
        T_0_init = self.T_0
        P_curr = self.P_0
        T_curr = self.T_0
        V_curr = self.len_inlet * (np.pi * (self.r_inlet**2))
        rho_curr = PropsSI('D', 'P', P_curr, 'T', T_curr, self.fluid)
        m_curr = V_curr * rho_curr
        h_curr = PropsSI('H', 'T', T_curr, 'P', P_curr, self.fluid)
        u_curr = PropsSI('U', 'T', T_curr, 'P', P_curr, self.fluid)
        H_curr = h_curr * m_curr
        U_curr = u_curr * m_curr
        C_d = 1
        
        AS = AbstractState("HEOS", self.fluid)

        # Adiabatic Blowdown
        while m_curr > 0:
            t += time_step
            m_curr = m_curr - m_dot_curr * time_step
            c_0 = np.sqrt(self.k*self.R_spec * T_0_init)

            exp = (self.k + 1) / (2 * self.k - 2)
            tau = (V_curr / (C_d * self.A_star * c_0))*(((self.k+1)/2)**exp)
            dPdt = - (self.k * P_0_init / tau) * (1 + ((self.k - 1) / 2) * (t / tau))**((2 * self.k) / (1 - self.k) - 1)
            dP = dPdt * time_step
            P_curr += dP
            self.canvas.axes_depress.scatter(t, P_curr)
            print(P_curr)
            print(m_curr)

        # while m_curr > 0:
        #     phase = PhaseSI('P', P_curr, 'T', T_curr, self.fluid)
        #     print(m_curr)
        #     print(phase)
        #     if phase == 'gas':
        #         m_curr = m_curr - m_dot_curr * time_step
        #         rho_curr = m_curr / V_curr
        #         U_curr = U_curr - ((m_dot_curr * time_step) * (h_curr))
        #         u_curr = U_curr / m_curr

        #         try:
        #             AS.update(CP.DmassUmass_INPUTS, rho_curr, u_curr)
        #         except ValueError:
        #             print(f'ValueError: Current density {rho_curr:.3g} is below triple')

        #         T_curr = AS.T()   # K
        #         P_curr = AS.p()   # Pa
        #         #h_curr = AS.h()   # J/kg

        #         # P_curr = PropsSI('P', 'T', T, 'D', rho_curr, self.fluid)
        #         # T_curr = PropsSI('T', 'T', T, 'D', rho_curr, self.fluid)
        #         h_curr = PropsSI('H', 'T', T_curr, 'D', rho_curr, self.fluid)

        #         # P_curr = PropsSI('P', 'u', u_curr, 'D', rho_curr, self.fluid)
        #         # T_curr = PropsSI('T', 'u', u_curr, 'D', rho_curr, self.fluid)
        #         # h_curr = PropsSI('H', 'u', u_curr, 'D', rho_curr, self.fluid)

        #         self.P_0 = P_curr
        #         self.T_0 = T_curr
        #         self.calc_thermo()
        #         m_dot_curr = self.m_dot
        #         self.canvas.axes_depress.scatter(i, P_curr)
        #         QApplication.processEvents()
        #         print(f'P = {P_curr}, T = {T_curr}')
        #         i+=1
        #     else:
        #         break
            
    def calc_opt_geom(self):
        """
        Calculates the optimal nozzle geometry using a gradient descent
        algorithm with an ADAM optimizer to match design thrust.
        """

        max_iterations=1000
        thr_design = float(self.thrust_design_val.text())
        area_ratio_outlet = self.A_outlet / self.A_star
        area_ratio_inlet = self.A_inlet / self.A_star
        tol=thr_design/10000            # Tolerance for convergence
        optimizer = ADAM_Optimizer(learning_rate=1E-3)

        def calc_cost():
            """
            Calculates the cost function to be minimized

            Returns:
                (float) - Cost function
            """
            self.calc_thermo()
            thr_curr = self.thr
            return (thr_curr - thr_design)**2

        def calc_gradient(delta=.0001):
            """
            Calculates the gradient of thrust with respect to throat radius 
            dT/dr_throat.

            Inputs:
                delta (float) - Small increment to throat radius for numerical
                differentiation
            
            Returns:
                (float) - Gradient of thrust with respect to throat radius
            """
            self.r_throat += delta
            self.r_outlet = ((self.r_throat**2) * area_ratio_outlet)**0.5
            self.r_inlet = ((self.r_throat**2) * area_ratio_inlet)**0.5
            
            cost_plus = calc_cost()
            self.r_throat -= delta
            self.r_outlet = ((self.r_throat**2) * area_ratio_outlet)**0.5
            self.r_inlet = ((self.r_throat**2) * area_ratio_inlet)**0.5
            cost_minus = calc_cost()
            return (cost_plus - cost_minus) / (2*delta)            

        for iteration in range(max_iterations):
            cost_curr = calc_cost()

            if abs(cost_curr) < tol:
                self.optimize_button.setText("Optimization Successful!")
                self.plot_data()
                break

            gradient = calc_gradient()
            update = optimizer.update(gradient)

            self.r_throat -= update
            self.r_outlet = ((self.r_throat**2) * area_ratio_outlet)**0.5
            self.r_inlet = ((self.r_throat**2) * area_ratio_inlet)**0.5
            self.radius_throat_val.setText("{:.4g}".format(self.r_throat))
            self.radius_exit_val.setText("{:.4g}".format(self.r_outlet))
            self.radius_inlet_val.setText("{:.4g}".format(self.r_inlet))
            self.optimize_button.setText("Optimizing... Itr: " + str(
                iteration))

            QApplication.processEvents()

        else:
            print(f"Convergence failed after {max_iterations:.0f} iterations")
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
