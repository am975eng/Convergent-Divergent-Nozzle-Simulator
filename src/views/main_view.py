from pathlib import Path

from PyQt6.QtCore import Qt
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
    QPushButton,
    QProgressBar,
)
from PyQt6.QtGui import QDoubleValidator
from dataclasses import dataclass
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.collections import LineCollection
import numpy as np
import scipy.stats as ss


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
    M_exit_moc: float
    thr_design: float
    noz_type: str
    depress_type: str
    monte_carlo_type: str


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
        self.resize(1500, 1000)
        screen = self.screen() or QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()

        # Get window geometry
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())

        # Move the top-left point of the window to the adjusted position
        self.move(window_geometry.topLeft())

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
        self.noz_type_list.addItems(
            [
                "MOC Full Length Nozzle",
                "MOC Minimum Length Nozzle",
                "Conical Nozzle",
                "Rao Bell Nozzle",
            ]
        )
        P_chamber_label = QLabel("Chamber Pressure [Pa]")
        P_chamber_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_chamber_val = QLineEdit("5000")
        self.P_chamber_val.setValidator(QDoubleValidator(0, 1e12, 3))
        T_chamber_label = QLabel("Chamber Temperature [K]")
        T_chamber_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.T_chamber_val = QLineEdit("293.15")
        self.T_chamber_val.setValidator(QDoubleValidator(0, 1e8, 3))
        converg_label = QLabel("Convergence Angle [Deg]")
        converg_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.converg_ang_val = QLineEdit("45")
        self.converg_ang_val.setValidator(QDoubleValidator(0, 360, 3))
        diverg_label = QLabel("Divergence Angle [Deg]")
        diverg_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.diverg_angle_val = QLineEdit("15")
        self.diverg_angle_val.setValidator(QDoubleValidator(0, 360, 3))
        thrust_design_label = QLabel("Design Thrust [N]")
        thrust_design_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.thrust_design_val = QLineEdit("100")
        self.thrust_design_val.setValidator(QDoubleValidator(0, 1e12, 3))
        self.optimize_button = QPushButton("Optimize Geometry")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.monte_carlo_type = QComboBox()
        self.monte_carlo_type.addItems(
            [
                "Chamber Pressure",
                "Ambient Pressure",
                "Throat Radius",
                "Outlet Radius"
            ]
        )

        length_inlet_label = QLabel("Inlet Length [m]")
        length_inlet_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.length_inlet_val = QLineEdit("0.1")
        self.length_inlet_val.setValidator(QDoubleValidator(0, 1e6, 6))
        radius_inlet_label = QLabel("Inlet Radius [m]")
        radius_inlet_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_inlet_val = QLineEdit("0.1")
        self.radius_inlet_val.setValidator(QDoubleValidator(0, 1e6, 6))
        self.radius_throat_label = QLabel("Throat Radius [m]")
        self.radius_throat_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_throat_val = QLineEdit("0.08")
        self.radius_throat_val.setValidator(QDoubleValidator(0, 1e6, 6))
        self.radius_outlet_label = QLabel("Exit Radius [m]")
        self.radius_outlet_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.radius_outlet_val = QLineEdit("0.1")
        self.radius_outlet_val.setValidator(QDoubleValidator(0, 1e6, 6))
        M_exit_moc_label = QLabel("Exit Mach Number")
        M_exit_moc_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.M_exit_moc_val = QLineEdit("2.0")
        self.M_exit_moc_val.setValidator(QDoubleValidator(1.001, 20, 5))
        P_amb_label = QLabel("Ambient Pressure [Pa]")
        P_amb_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.P_amb_val = QLineEdit("0")
        self.P_amb_val.setValidator(QDoubleValidator(0, 1e12, 3))
        depress_type = QLabel("Depressurization Type")
        depress_type.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.depress_type_list = QComboBox()
        self.depress_type_list.addItems(["Isothermal", "Adiabatic"])
        self.depress_button = QPushButton("Depressurize")
        self.monte_carlo_button = QPushButton("Run Monte Carlo")
        self.calc_button = QPushButton("Run")

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
        self.P_e_val = QLabel(" ")
        self.P_e_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
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
            option_layout.addWidget(widget, i + start, col)
            for i, widget in enumerate(widgets)
        ]
        add_to_layout(
            0,
            0,
            DC_label,
            prop_label,
            self.prop_list,
            noz_label,
            self.noz_type_list,
            P_chamber_label,
            self.P_chamber_val,
            T_chamber_label,
            self.T_chamber_val,
            converg_label,
            self.converg_ang_val,
            diverg_label,
            self.diverg_angle_val,
            thrust_design_label,
            self.thrust_design_val,
            self.optimize_button,
            self.monte_carlo_type
        )
        add_to_layout(
            1,
            1,
            length_inlet_label,
            self.length_inlet_val,
            radius_inlet_label,
            self.radius_inlet_val,
            self.radius_throat_label,
            self.radius_throat_val,
            self.radius_outlet_label,
            self.radius_outlet_val,
            M_exit_moc_label,
            self.M_exit_moc_val,
            P_amb_label,
            self.P_amb_val,
            depress_type,
            self.depress_type_list,
            self.depress_button,
            self.monte_carlo_button
        )
        v_spacer = QSpacerItem(
            20,
            40,
            QSizePolicy.Policy.Minimum,  # Doesn’t expand sideways
            QSizePolicy.Policy.Expanding,  # Absorbs all extra vertical space
        )
        h_spacer = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        option_layout.addWidget(self.progress_bar, 17, 0, 1, 1)
        option_layout.addWidget(self.calc_button, 17, 1, 1, 1)
        option_layout.addItem(v_spacer, 18, 0, 1, 2)

        option_layout.addWidget(results_label, 19, 0)
        option_layout.addWidget(self.result_display_label, 20, 0, 1, 2)
        add_to_layout(
            0,
            21,
            P_e_sup_label,
            self.P_e_sup_val,
            P_e_shock_label,
            self.P_e_shock_val,
            m_dot_label,
            self.m_dot_val,
            m_prop_label,
            self.m_prop_val,
            ISP_label,
            self.ISP_val,
        )
        add_to_layout(
            1,
            21,
            P_e_sub_label,
            self.P_e_sub_val,
            P_star_shock_label,
            self.P_star_shock_val,
            P_exit_label,
            self.P_e_val,
            expansion_ratio_label,
            self.expansion_ratio_val,
            thrust_label,
            self.thrust_val,
        )

        graphic_layout = QVBoxLayout()
        self.canvas = MplCanvas(self)
        graphic_layout.addWidget(self.canvas)

        outer_layout = QHBoxLayout()
        outer_layout.addLayout(option_layout)
        outer_layout.addSpacerItem(h_spacer)
        outer_layout.addLayout(graphic_layout)

        widget.setLayout(outer_layout)
        self.setCentralWidget(widget)

        self.grey_out_style = """QLineEdit[enabled="false"]
            {Background-color: #a3a3a3; color: white;}
            QLineEdit[enabled="true"]
            {Background-color: white; color: black;}"""

        self.canvas.axes.set_title(
            "Centerline Values", color="white", fontsize=10
        )
        self.canvas.axes.set_ylabel("Y Position [m]", color="white")
        self.canvas.axes_mass.set_title(
            "Depressurization Values", color="white", fontsize=10
        )
        self.canvas.axes_mass.set_ylabel("Mass [kg]", color="white")
        self.canvas.axes_press.set_ylabel("Pressure [Pa]", color="white")
        self.canvas.axes_depress.set_ylabel("Pressure [Pa]", color="white")
        self.canvas.axes_mach.set_ylabel("Mach Number", color="white")
        self.canvas.axes_thrust.set_ylabel("Thrust [N]", color="white")
        self.canvas.axes_temp.set_ylabel("Temperature [K]", color="white")
        self.canvas.axes_temp.set_xlabel("X Position [m]", color="white")
        self.canvas.axes_detemp.set_ylabel("Temperature [K]", color="white")
        self.canvas.axes_detemp.set_xlabel("Time [s]", color="white")

    def extract_UI_data(self):
        """
        Extracts UI data from input widgets.

        Returns:
            UIInputs (dataclass) - Contains any data input by user.
        """

        fluid = self.prop_list.currentText()
        if fluid == "Air":
            R_spec = 287
            k = 1.4
        elif fluid == "CO2":
            R_spec = 188.9
            k = 1.289
        elif fluid == "N2":
            R_spec = 296.8
            k = 1.4
        elif fluid == "Xe":
            R_spec = 63.33
            k = 1.667

        # Extract thermo. properties
        T_0 = float(self.T_chamber_val.text())
        P_0 = float(self.P_chamber_val.text())
        rho_0 = P_0 / (R_spec * T_0)
        P_amb = float(self.P_amb_val.text())

        # Throat Conditions
        T_star = T_0 * (2 / (k + 1))
        P_star = P_0 * ((2 / (k + 1)) ** (k / (k - 1)))
        M_star = 1

        # Geometry
        r_throat = float(self.radius_throat_val.text())
        r_inlet = float(self.radius_inlet_val.text())
        r_outlet = float(self.radius_outlet_val.text())
        converg_angle = np.deg2rad(float(self.converg_ang_val.text()))
        diverg_angle = np.deg2rad(float(self.diverg_angle_val.text()))
        len_inlet = float(self.length_inlet_val.text())

        M_exit_moc = float(self.M_exit_moc_val.text())
        thr_design = float(self.thrust_design_val.text())

        noz_type = self.noz_type_list.currentText()
        depress_type = self.depress_type_list.currentText()
        monte_carlo_type = self.monte_carlo_type.currentText()

        return UIInputs(
            fluid,
            R_spec,
            k,
            T_0,
            P_0,
            rho_0,
            P_amb,
            T_star,
            P_star,
            M_star,
            r_throat,
            r_inlet,
            r_outlet,
            converg_angle,
            diverg_angle,
            len_inlet,
            M_exit_moc,
            thr_design,
            noz_type,
            depress_type,
            monte_carlo_type
        )

    def update_UI_nozzle(self):
        if (
            self.noz_type_list.currentText() == "MOC Full Length Nozzle"
            or self.noz_type_list.currentText() == "MOC Minimum Length Nozzle"
        ):
            self.converg_ang_val.setEnabled(False)
            self.converg_ang_val.setStyleSheet(self.grey_out_style)
            self.diverg_angle_val.setEnabled(False)
            self.diverg_angle_val.setStyleSheet(self.grey_out_style)
            self.radius_outlet_val.setEnabled(False)
            self.radius_outlet_val.setStyleSheet(self.grey_out_style)
            self.M_exit_moc_val.setEnabled(True)
            self.M_exit_moc_val.setStyleSheet(self.grey_out_style)
        elif (
            self.noz_type_list.currentText() == "Conical Nozzle"
            or self.noz_type_list.currentText() == "Rao Bell Nozzle"
        ):
            self.converg_ang_val.setEnabled(True)
            self.converg_ang_val.setStyleSheet(self.grey_out_style)
            self.diverg_angle_val.setEnabled(True)
            self.diverg_angle_val.setStyleSheet(self.grey_out_style)
            self.radius_outlet_val.setEnabled(True)
            self.radius_outlet_val.setStyleSheet(self.grey_out_style)
            self.M_exit_moc_val.setEnabled(False)
            self.M_exit_moc_val.setStyleSheet(self.grey_out_style)

    def print_results(self, flow_result):
        """Update UI display table with results."""

        # Result Section
        self.result_display_label.setText(flow_result.result_display)
        self.P_e_sup_val.setText("{:.3g}".format(flow_result.P_e_sup))
        self.P_e_sub_val.setText("{:.3g}".format(flow_result.P_e_sub))
        self.P_e_shock_val.setText("{:.3g}".format(flow_result.P_e_shock))
        self.P_star_shock_val.setText("{:.3g}".format(flow_result.P_star_shock))
        self.m_dot_val.setText("{:.3g}".format(flow_result.m_dot))
        self.P_e_val.setText("{:.3g}".format(flow_result.P_e))
        self.m_prop_val.setText("{:.3g}".format(flow_result.m_prop))
        self.expansion_ratio_val.setText(
            "{:.3g}".format(flow_result.expansion_ratio)
        )
        self.ISP_val.setText("{:.3g}".format(flow_result.ISP))
        self.thrust_val.setText("{:.4g}".format(flow_result.thr))

    def plot_flow_data(self, UI_input, flow_result, bar_value=None):
        """Updates flow plots with new data.

        Inputs:
            UI_input (dataclass) - User inputs
            flow_result (dataclass) - Flow results
        """

        def reflect_plot(axes):
            """
            Reflects the plot on the opposite side of the x-axis

            Inputs:
                axes (matplotlib axis) - Axis to plot on
            """
            lines = axes.get_lines()
            for line in lines:
                x_data = line.get_xdata()
                y_data = line.get_ydata()

                axes.plot(
                    x_data,
                    -y_data,
                    color=line.get_color(),
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                )

        def gen_cmap_plot(x, y, ax):
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
            ax.set_ylim(np.min(y) - 0.1 * span, np.max(y) + 0.15 * span)

        # Clear plots
        for axes in [
            self.canvas.axes,
            self.canvas.axes_mass,
            self.canvas.axes_press,
            self.canvas.axes_depress,
            self.canvas.axes_mach,
            self.canvas.axes_thrust,
            self.canvas.axes_temp,
            self.canvas.axes_detemp,
        ]:
            for line in axes.lines:
                line.remove()
            for collection in axes.collections:
                collection.remove()

        self.canvas.axes.plot(
            [flow_result.x_conv[0], flow_result.x_conv[0] - UI_input.len_inlet],
            [flow_result.y_conv[0], flow_result.y_conv[0]],
            "b-",
            linewidth=2,
        )
        self.canvas.axes.plot(
            [
                flow_result.x_conv[0] - UI_input.len_inlet,
                flow_result.x_conv[0] - UI_input.len_inlet,
            ],
            [flow_result.y_conv[0], 0],
            "b-",
            linewidth=2,
        )
        self.canvas.axes.plot(
            flow_result.x_conv, flow_result.y_conv, "b-", linewidth=2
        )
        self.canvas.axes.plot(
            flow_result.x_div, flow_result.y_div, "b-", linewidth=2
        )

        gen_cmap_plot(
            np.concatenate([flow_result.x_conv, flow_result.x_div]),
            flow_result.P_array,
            self.canvas.axes_press,
        )
        gen_cmap_plot(
            np.concatenate([flow_result.x_conv, flow_result.x_div]),
            flow_result.M_array,
            self.canvas.axes_mach,
        )
        gen_cmap_plot(
            np.concatenate([flow_result.x_conv, flow_result.x_div]),
            flow_result.T_array,
            self.canvas.axes_temp,
        )

        # Plot shockwaves
        if flow_result.x_under_shock[0] is not None:
            for i in range(len(flow_result.x_under_shock[1])):
                self.canvas.axes.plot(
                    [
                        flow_result.x_under_shock[0],
                        flow_result.x_under_shock[1][i],
                    ],
                    [
                        flow_result.y_under_shock[0],
                        flow_result.y_under_shock[1][i],
                    ],
                    "g-",
                )
        elif flow_result.x_over_shock[0] is not None:
            self.canvas.axes.plot(
                flow_result.x_over_shock, flow_result.y_over_shock, "r-"
            )

        self.canvas.axes.relim()
        self.canvas.axes.autoscale()

        reflect_plot(self.canvas.axes)

        self.print_results(flow_result)
        if bar_value is not None:
            self.progress_bar.setValue(bar_value)
            self.radius_throat_val.setText("{:.3g}".format(UI_input.r_throat))
            self.radius_outlet_val.setText("{:.3g}".format(UI_input.r_outlet))
            self.radius_inlet_val.setText("{:.3g}".format(UI_input.r_inlet))

        # Refresh canvas
        self.canvas.draw_idle()

    def plot_depress_update(self, depress_data_update):
        """
        Plots the current state of depressurization.

        Inputs:
            depress_data_update (tuple) - Tuple containing the following:
                UI_input (dataclass) - User inputs
                flow_result (dataclass) - Flow results
                t (float) - Current time
                P_curr (float) - Current pressure
                m_curr (float) - Current mass
                thr_curr (float) - Current thrust
                T_curr (float) - Current temperature
                i (float) - Current iteration num/max_iterations
        """
        UI_input, flow_result, t, P_curr, m_curr, thr_curr, T_curr, i = (
            depress_data_update
        )
        self.print_results(flow_result)

        self.progress_bar.setValue(int(i * 100))

        self.canvas.axes_depress.scatter(t, P_curr, color="g")
        self.canvas.axes_mass.scatter(t, m_curr, color="r")
        self.canvas.axes_thrust.scatter(t, thr_curr, color="b")
        self.canvas.axes_detemp.scatter(t, T_curr, color="w")

        self.canvas.draw_idle()

    def plot_depress_final(self, depress_result):
        """Plots the complete depressurization curve and updates flow
        centerline plots.

        Inputs:
            depress_result (tuple) - Tuple containing the following:
                UI_input (dataclass) - User inputs
                flow_result (dataclass) - Flow results
                t_depress_array (np.array) - Time array
                P_depress_array (np.array) - Pressure array
                m_depress_array (np.array) - Mass array
                thr_depress_array (np.array) - Thrust array
                temp_depress_array (np.array) - Temperature array
        """
        (
            UI_input,
            flow_result,
            t_depress_array,
            P_depress_array,
            m_depress_array,
            thr_depress_array,
            temp_depress_array,
        ) = depress_result

        self.plot_flow_data(UI_input, flow_result)

        self.canvas.axes_depress.plot(t_depress_array, P_depress_array, "g-")
        self.canvas.axes_mass.plot(t_depress_array, m_depress_array, "r-")
        self.canvas.axes_thrust.plot(t_depress_array, thr_depress_array, "b-")
        self.canvas.axes_detemp.plot(t_depress_array, temp_depress_array, "w-")

        self.canvas.draw_idle()

    def set_busy_state(self, type):
        """Sets the busy state of the UI to prevent user interaction."""
        if type == "optimize":
            self.calc_button.setStyleSheet("background-color: blue;")
            self.optimize_button.setText("Optimizing...")
            self.optimize_button.setEnabled(False)
            self.depress_button.setEnabled(False)
            self.monte_carlo_button.setEnabled(False)
            self.calc_button.setEnabled(False)
            self.progress_bar.setValue(0)
        elif type == "depress":
            self.depress_button.setText("Depressing...")
            self.optimize_button.setEnabled(False)
            self.depress_button.setEnabled(False)
            self.monte_carlo_button.setEnabled(False)
            self.calc_button.setEnabled(False)
            self.progress_bar.setValue(0)
        elif type == "monte_carlo":
            self.monte_carlo_button.setText("Simulating...")
            self.monte_carlo_button.setEnabled(False)
            self.optimize_button.setEnabled(False)
            self.depress_button.setEnabled(False)
            self.monte_carlo_button.setEnabled(False)
            self.calc_button.setEnabled(False)
        elif type == "finished":
            self.optimize_button.setText("Optimize Geometry")
            self.optimize_button.setEnabled(True)
            self.depress_button.setText("Depressurize")
            self.depress_button.setEnabled(True)
            self.progress_bar.setValue(100)
            self.monte_carlo_button.setEnabled(True)
            self.monte_carlo_button.setText("Run Monte Carlo")
            self.calc_button.setEnabled(True)
            self.calc_button.setStyleSheet("background-color: green;")

    def plot_monte_carlo(self, mc_thrust_array):
        """Plots Monte Carlo results including thrust histogram and a normal
        PDF fit.

        Inputs:
            mc_thrust_array (np.array) - Array of thrust samples
        """
        mu, sigma = ss.norm.fit(mc_thrust_array)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
        pdf = ss.norm.pdf(x, mu, sigma)

        plt.figure()
        plt.plot(x, pdf, 'r-', linewidth=2, label="Normal PDF fit")
        plt.hist(mc_thrust_array, density=True, label="Histogram")
        plt.xlabel("Thrust [N]")
        plt.ylabel("Probability Density")
        plt.title("Monte Carlo Simulation of Thrust")
        plt.show()
            
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
        self.fig = Figure(figsize=(9, 9), constrained_layout=True)
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
        self.axes.title.set_color("white")

        all_axes = [
            self.axes,
            self.axes_mass,
            self.axes_press,
            self.axes_depress,
            self.axes_mach,
            self.axes_thrust,
            self.axes_temp,
            self.axes_detemp,
        ]
        for ax in all_axes:
            ax.set_facecolor("none")
            ax.spines["top"].set_color("white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["right"].set_color("white")
            ax.tick_params(colors="white")

        super().__init__(self.fig)
        self.setParent(parent)
