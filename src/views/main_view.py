from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QComboBox, QGridLayout, QLabel,
                             QLineEdit, QMainWindow, QSpacerItem, QWidget,
                             QVBoxLayout, QHBoxLayout, QSizePolicy, 
                             QPushButton)
from dataclasses import dataclass
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure
from pathlib import Path
import numpy as np

@dataclass
class UIInputs:
    fluid: str
    R_spec: float
    k: float
    T_0: float
    P_0: float
    rho_0: float
    P_amb: float
    T_star: float
    P_star: float
    M_star: float
    r_throat: float
    r_inlet: float
    r_outlet: float
    converg_angle: float
    diverg_angle: float
    len_inlet: float
    M_exit: float
    thr_design: float
    noz_type: str

class MainWindow(QMainWindow):
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
            style_file = current_dir.parent / "style.qss"

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
        noz_label = QLabel("Nozzle Geometry Type")
        noz_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.noz_type_list = QComboBox()
        self.noz_type_list.addItems(["MOC Full Length Nozzle",
                                     "MOC Minimum Length Nozzle", "Conical"])
        P_chamber_label = QLabel("Chamber Pressure [Pa]")
        P_chamber_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_chamber_val = QLineEdit("5000")
        T_chamber_label = QLabel("Chamber Temperature [K]")
        T_chamber_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.T_chamber_val = QLineEdit("293.15")
        converg_label = QLabel("Convergence Angle [Deg]")
        converg_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.converg_ang_val = QLineEdit("45")
        diverg_label = QLabel("Divergence Angle [Deg]")
        diverg_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.diverg_angle_val = QLineEdit("15")
        thrust_design_label = QLabel("Design Thrust [N]")
        thrust_design_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.thrust_design_val = QLineEdit("100")
        self.optimize_button = QPushButton("Optimize Geometry")

        length_inlet_label = QLabel("Inlet Length [m]")
        length_inlet_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.length_inlet_val = QLineEdit("0.1")
        radius_inlet_label = QLabel("Inlet Radius [m]")
        radius_inlet_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_inlet_val = QLineEdit("0.1")
        self.radius_throat_label = QLabel("Throat Radius [m]")
        self.radius_throat_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_throat_val = QLineEdit("0.08")
        self.radius_exit_label = QLabel("Exit Radius [m]")
        self.radius_exit_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_exit_val = QLineEdit("0.1")
        M_exit_label = QLabel("Exit Mach Number")
        M_exit_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.M_exit_val = QLineEdit("2.0")
        P_amb_label = QLabel("Ambient Pressure [Pa]")
        P_amb_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_amb_val = QLineEdit("0")
        depress_type = QLabel("Depressurization Type")
        depress_type.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.depress_type_list = QComboBox()
        self.depress_type_list.addItems(["Isothermal", "Adiabatic"])
        self.depress_button = QPushButton("Depressurize")

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
        m_prop_label = QLabel("Propellant Mass [kg]")
        m_prop_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.m_prop_val = QLabel(" ")
        self.m_prop_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        expansion_ratio_label = QLabel("Expansion Ratio")
        expansion_ratio_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.expansion_ratio_val = QLabel(" ")
        self.expansion_ratio_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
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
                      self.diverg_angle_val, thrust_design_label,
                      self.thrust_design_val, self.optimize_button)
        add_to_layout(1, 1, length_inlet_label, self.length_inlet_val, 
                      radius_inlet_label, self.radius_inlet_val, 
                      self.radius_throat_label, self.radius_throat_val,
                      self.radius_exit_label, self.radius_exit_val,
                      M_exit_label, self.M_exit_val, P_amb_label,
                      self.P_amb_val, depress_type, self.depress_type_list,
                      self.depress_button)
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
                      m_prop_label, self.m_prop_val,ISP_label, self.ISP_val)
        add_to_layout(1,19, P_e_sub_label, self.P_e_sub_val,
                      P_star_shock_label, self.P_star_shock_val, P_exit_label,
                      self.P_exit_val, expansion_ratio_label,
                      self.expansion_ratio_val, thrust_label, self.thrust_val)

        graphic_layout = QVBoxLayout()
        self.canvas = MplCanvas(self)
        graphic_layout.addWidget(self.canvas)

        outer_layout = QHBoxLayout()
        outer_layout.addLayout(option_layout)
        outer_layout.addSpacerItem(h_spacer)
        outer_layout.addLayout(graphic_layout)

        widget.setLayout(outer_layout)
        self.setCentralWidget(widget)

        self.canvas.axes.set_title('Centerline Values', color='white', 
                                   fontsize=10)
        self.canvas.axes.set_ylabel('Y Position [m]', color='white')
        self.canvas.axes_mass.set_title('Depressurization Values',
                                        color='white', fontsize=10)
        self.canvas.axes_mass.set_ylabel('Mass [kg]', color='white')
        self.canvas.axes_press.set_ylabel('Pressure [Pa]', color='white')
        self.canvas.axes_depress.set_ylabel('Pressure [Pa]', color='white')
        self.canvas.axes_mach.set_ylabel('Mach Number', color='white')
        self.canvas.axes_thrust.set_ylabel('Thrust [N]', color='white')
        self.canvas.axes_temp.set_ylabel('Temperature [K]', color='white')
        self.canvas.axes_temp.set_xlabel('X Position [m]', color='white')
        self.canvas.axes_detemp.set_ylabel('Temperature [K]', color='white')
        self.canvas.axes_detemp.set_xlabel('Time [s]', color='white')

    def extract_UI_data(self):
        fluid = self.prop_list.currentText()
        if fluid == "Air":
            R_spec = 287
            k=1.4
        elif fluid == "CO2":
            R_spec = 188.9
            k=1.289
        elif fluid == "N2":
            R_spec = 296.8
            k=1.4
        elif fluid == "Xe":
            R_spec = 63.33
            k=1.667

        # Extract thermo. properties
        T_0 = float(self.T_chamber_val.text())
        P_0 = float(self.P_chamber_val.text())
        rho_0 = P_0/(R_spec*T_0)
        P_amb = float(self.P_amb_val.text())

        # Throat Conditions
        T_star = T_0 * (2/(k+1))
        P_star = P_0 * ((2/(k+1))**(k/(k-1)))
        M_star = 1

        # Geometry
        r_throat = float(self.radius_throat_val.text())
        r_inlet = float(self.radius_inlet_val.text())
        r_outlet = float(self.radius_exit_val.text())
        converg_angle= np.deg2rad(float(self.converg_ang_val.text()))
        diverg_angle = np.deg2rad(float(self.diverg_angle_val.text()))
        len_inlet = float(self.length_inlet_val.text())

        M_exit = float(self.M_exit_val.text())
        thr_design = float(self.thrust_design_val.text())

        noz_type = self.noz_type_list.currentText()

        return UIInputs(fluid, R_spec, k, T_0, P_0, rho_0, P_amb, T_star,
                        P_star, M_star, r_throat, r_inlet, r_outlet,
                        converg_angle, diverg_angle, len_inlet, M_exit,
                        thr_design, noz_type)


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
        self.axes = self.fig.add_subplot(421)
        self.axes_mass = self.fig.add_subplot(422)
        self.axes_press = self.fig.add_subplot(423)
        self.axes_depress = self.fig.add_subplot(424)
        self.axes_mach = self.fig.add_subplot(425)
        self.axes_thrust = self.fig.add_subplot(426)
        self.axes_temp = self.fig.add_subplot(427)
        self.axes_detemp = self.fig.add_subplot(428)

        self.axes.grid(True)
        self.axes.title.set_color('white')

        all_axes = [self.axes, self.axes_mass, self.axes_press, 
                    self.axes_depress, self.axes_mach, self.axes_thrust, 
                    self.axes_temp, self.axes_detemp]
        for ax in all_axes:
            ax.set_facecolor('none')
            ax.spines['top'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.tick_params(colors='white')

        super().__init__(self.fig)
        self.setParent(parent)