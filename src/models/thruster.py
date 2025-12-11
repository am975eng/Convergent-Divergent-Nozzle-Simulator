import math
import time

import numpy as np
from dataclasses import dataclass
from scipy.optimize import fsolve, root_scalar
from scipy.stats import norm, uniform
from CoolProp.CoolProp import PropsSI, AbstractState

from thermodynamics import aero_thermo as AT
from thermodynamics import method_of_char as MOC
from optimizers.optimizers import ADAM_Optimizer
from logger import setup_logger


@dataclass
class FlowResults:
    x_conv: np.ndarray
    y_conv: np.ndarray
    x_div: np.ndarray
    y_div: np.ndarray
    P_array: np.ndarray
    M_array: np.ndarray
    T_array: np.ndarray
    P_e_sup: float
    P_e_sub: float
    P_e_shock: float
    P_star_shock: float
    m_dot: float
    P_e: float
    expansion_ratio: float
    m_prop: float
    ISP: float
    thr: float
    result_display: str
    x_under_shock: list
    y_under_shock: list
    x_over_shock: list
    y_over_shock: list


logger = setup_logger(__name__)


class ThrusterModel:
    """
    Thermodynamic model of a thruster with methods for calculating optimal
    geometry and depressurization conditions.
    """

    def calc_thermo(self, UI_input, res=150):
        """
        Calculates flow properties throughout nozzle centerline.

        Inputs:
            UI_input (dataclass) - User inputs
        Outputs:
            flow_result (dataclass) - Flow results
        """
        logger.info("Calculating flow properties...")
        A_star = math.pi * (UI_input.r_throat**2)
        A_inlet = math.pi * (UI_input.r_inlet**2)
        A_outlet = math.pi * (UI_input.r_outlet**2)
        converg_length = (UI_input.r_inlet - UI_input.r_throat) / np.tan(
            UI_input.converg_angle
        )
        diverg_length = (UI_input.r_outlet - UI_input.r_throat) / np.tan(
            UI_input.diverg_angle
        )
        x_conv = -np.flip(np.linspace(0, converg_length, res))
        y_conv = UI_input.r_throat - x_conv * np.tan(UI_input.converg_angle)

        if UI_input.noz_type == "MOC Full Length Nozzle":
            x_div, y_div, *_ = MOC.gen_MOC_FLN(
                UI_input.M_exit_moc, UI_input.r_throat, UI_input.k, div=50
            )
            UI_input.r_outlet = np.max(y_div)
            A_outlet = math.pi * (UI_input.r_outlet**2)
        elif UI_input.noz_type == "MOC Minimum Length Nozzle":
            x_div, y_div, *_ = MOC.gen_MOC_MLN(
                UI_input.M_exit_moc, UI_input.r_throat, UI_input.k, div=50
            )
            UI_input.r_outlet = np.max(y_div)
            A_outlet = math.pi * (UI_input.r_outlet**2)
        elif UI_input.noz_type == "Conical Nozzle":
            x_div = np.linspace(0, diverg_length, res)
            y_div = UI_input.r_throat + x_div * np.tan(UI_input.diverg_angle)
        elif UI_input.noz_type == "Rao Bell Nozzle":
            x_div, y_div = MOC.gen_rao_bell(
                UI_input.r_throat, UI_input.r_outlet, k=UI_input.k, len_per=80
            )

        # Due to fp math div sect may be less than r_throat
        if UI_input.r_throat < np.min(y_div):
            y_div = y_div + ((UI_input.r_throat - np.min(y_div)) * 1.5)

        try:
            M_e_sup = root_scalar(
                AT.RS_Area_Mach_X_Y,
                bracket=[1, 100],
                args=(UI_input.M_star, A_outlet, A_star, UI_input.k),
            ).root
            M_e_sub = root_scalar(
                AT.RS_Area_Mach_X_Y,
                bracket=[0.0001, 1],
                args=(UI_input.M_star, A_outlet, A_star, UI_input.k),
            ).root
        except ValueError as e:
            logger.error(
                "Failed to calculate Mach numbers at exit, expand root_scalar solution bracket to solve."
            )

        # Preallocation
        P_array = np.zeros(len(x_conv) + len(x_div))
        M_array = np.zeros(len(x_conv) + len(x_div))
        T_array = np.zeros(len(x_conv) + len(x_div))

        P_e_sup = AT.calc_isen_press(M_e_sup, UI_input.P_0, UI_input.k)
        P_e_sub = AT.calc_isen_press(M_e_sub, UI_input.P_0, UI_input.k)

        x_shock = None  # Shock location
        x_under_shock_1, y_under_shock_1 = None, None
        x_under_shock_2, y_under_shock_2 = None, None
        x_over_shock_1, y_over_shock_1 = None, None
        x_over_shock_2, y_over_shock_2 = None, None

        # Solve for shock at exit of nozzle
        A_x = math.pi * (y_div[-1] ** 2)
        # Mach number before shock
        M_x_sup = root_scalar(
            AT.RS_Area_Mach_X_Y,
            bracket=[1, 100],
            args=(UI_input.M_star, A_x, A_star, UI_input.k),
        ).root
        P_x = AT.calc_isen_press(M_x_sup, UI_input.P_0, UI_input.k)
        T_x = AT.calc_isen_temp(M_x_sup, UI_input.T_0, UI_input.k)
        M_y, P_y = AT.calc_M_P_normal(M_x_sup, P_x, UI_input.k)
        UI_input.P_0_y = AT.calc_isen_stag_press(M_y, P_y, UI_input.k)
        UI_input.T_0_y = UI_input.T_0

        # New critical area using post shocUI_input.k conditions
        A_star_shock = (
            A_x
            * M_y
            * (
                (
                    (2 / (UI_input.k + 1))
                    * (1 + ((UI_input.k - 1) / 2) * M_y * M_y)
                )
                ** ((-UI_input.k - 1) / (2 * UI_input.k - 2))
            )
        )
        M_y_e = root_scalar(
            AT.RS_Area_Mach_X_Y,
            bracket=[0.00001, 1],
            args=(UI_input.M_star, A_outlet, A_star_shock, UI_input.k),
        ).root
        P_e_shock = AT.calc_isen_press(M_y_e, UI_input.P_0_y, UI_input.k)

        # Solve for shock at onset of divergent section
        A_x = math.pi * (y_div[1] ** 2)
        M_x_sup = root_scalar(
            AT.RS_Area_Mach_X_Y,
            bracket=[1, 100],
            args=(UI_input.M_star, A_x, A_star, UI_input.k),
        ).root
        P_x = AT.calc_isen_press(M_x_sup, UI_input.P_0, UI_input.k)
        T_x = AT.calc_isen_temp(M_x_sup, UI_input.T_0, UI_input.k)
        M_y, P_y = AT.calc_M_P_normal(M_x_sup, P_x, UI_input.k)
        UI_input.P_0_y = AT.calc_isen_stag_press(M_y, P_y, UI_input.k)
        UI_input.T_0_y = UI_input.T_0

        A_star_shock = (
            A_x
            * M_y
            * (
                (
                    (2 / (UI_input.k + 1))
                    * (1 + ((UI_input.k - 1) / 2) * M_y * M_y)
                )
                ** ((-UI_input.k - 1) / (2 * UI_input.k - 2))
            )
        )
        M_y_e = root_scalar(
            AT.RS_Area_Mach_X_Y,
            bracket=[0.0001, 1],
            args=(UI_input.M_star, A_outlet, A_star_shock, UI_input.k),
        ).root
        P_star_shock = AT.calc_isen_press(M_y_e, UI_input.P_0_y, UI_input.k)

        logger.info("Comparing back and exit pressure...")
        if abs(UI_input.P_amb - P_e_sup) / P_e_sup < 0.1 or (
            UI_input.P_amb < 50 and P_e_sup < 100
        ):
            # Check if Back pressure and supersonic exit pressure are close
            # enough for perfect expansion
            result_display = "Perfectly expanded supersonic exhaust!"
            logger.info(result_display)
            for index in range(len(x_div)):
                A_x = math.pi * (y_div[index] ** 2)
                shift = index + len(x_conv)
                M_x_sup = root_scalar(
                    AT.RS_Area_Mach_X_Y,
                    bracket=[1, 100],
                    args=(UI_input.M_star, A_x, A_star, UI_input.k),
                ).root
                P_x = AT.calc_isen_press(M_x_sup, UI_input.P_0, UI_input.k)
                T_x = AT.calc_isen_temp(M_x_sup, UI_input.T_0, UI_input.k)
                P_array[shift] = P_x
                M_array[shift] = M_x_sup
                T_array[shift] = T_x

        elif UI_input.P_amb >= UI_input.P_0:
            result_display = (
                "Back pressure high enough to generate reversed flow"
            )
            logger.error(result_display)
        elif UI_input.P_amb >= P_e_sub:
            # Back pressure is too high for choked subsonic flow
            result_display = "Back pressure too high for choked subsonic flow"
            logger.info(result_display)
            M_e_sub = root_scalar(
                AT.RS_Mach_Press_Isen,
                bracket=[0.0001, 1],
                args=(0, UI_input.P_amb, UI_input.P_0, UI_input.k),
            ).root
            UI_input.M_star = root_scalar(
                AT.RS_Area_Mach_X_Y,
                bracket=[0.0001, 1],
                args=(M_e_sub, A_star, A_outlet, UI_input.k),
            ).root

            for index in range(len(x_div)):
                A_x = math.pi * (y_div[index] ** 2)
                shift = index + len(x_conv)

                M_x = root_scalar(
                    AT.RS_Area_Mach_X_Y,
                    bracket=[0.0001, 1],
                    args=(M_e_sub, A_x, A_outlet, UI_input.k),
                ).root
                P_x = AT.calc_isen_press(M_x, UI_input.P_0, UI_input.k)
                T_x = AT.calc_isen_temp(M_x, UI_input.T_0, UI_input.k)

                P_array[shift] = P_x
                M_array[shift] = M_x
                T_array[shift] = T_x

        elif UI_input.P_amb > P_e_shock:
            logger.info(
                "Back pressure is low enough for choked supersonic flow with possible normal shock"
            )
            for index in range(len(x_div)):
                A_x = math.pi * (y_div[index] ** 2)
                shift = index + len(x_conv)
                if x_shock is None:
                    # Pre shock wave prop
                    M_x_sup = root_scalar(
                        AT.RS_Area_Mach_X_Y,
                        bracket=[1, 100],
                        args=(UI_input.M_star, A_x, A_star, UI_input.k),
                    ).root
                    P_x = AT.calc_isen_press(M_x_sup, UI_input.P_0, UI_input.k)
                    T_x = AT.calc_isen_temp(M_x_sup, UI_input.T_0, UI_input.k)
                    P_array[shift] = P_x
                    M_array[shift] = M_x_sup
                    T_array[shift] = T_x

                    # Post shock wave prop
                    M_y, P_y = AT.calc_M_P_normal(M_x_sup, P_x, UI_input.k)
                    UI_input.P_0_y = AT.calc_isen_stag_press(
                        M_y, P_y, UI_input.k
                    )
                    UI_input.T_0_y = UI_input.T_0

                    A_star_shock = (
                        A_x
                        * M_y
                        * (
                            (
                                (2 / (UI_input.k + 1))
                                * (1 + ((UI_input.k - 1) / 2) * M_y * M_y)
                            )
                            ** ((-UI_input.k - 1) / (2 * UI_input.k - 2))
                        )
                    )
                    M_y_e = root_scalar(
                        AT.RS_Area_Mach_X_Y,
                        bracket=[0.0001, 1],
                        args=(
                            UI_input.M_star,
                            A_outlet,
                            A_star_shock,
                            UI_input.k,
                        ),
                    ).root
                    P_y_e = AT.calc_isen_press(
                        M_y_e, UI_input.P_0_y, UI_input.k
                    )
                    T_y_e = UI_input.T_0_y * (
                        (1 + ((UI_input.k - 1) / 2) * M_y_e * M_y_e) ** -1
                    )

                    if abs((P_y_e - UI_input.P_amb) / P_y_e) < 0.01:
                        x_shock = x_div[index]
                        result_display = (
                            f"Normal shock generated in divergent section"
                            f"at {x_shock:.3f} m"
                        )
                        logger.info(result_display)
                else:
                    M_y_curr = root_scalar(
                        AT.RS_Area_Mach_X_Y,
                        bracket=[0.0001, 1],
                        args=(UI_input.M_star, A_x, A_star_shock, UI_input.k),
                    ).root
                    P_y_curr = AT.calc_isen_press(
                        M_y_curr, UI_input.P_0_y, UI_input.k
                    )
                    T_y_curr = AT.calc_isen_temp(
                        M_y_curr, UI_input.T_0_y, UI_input.k
                    )
                    P_array[shift] = P_y_curr
                    M_array[shift] = M_y_curr
                    T_array[shift] = T_y_curr

        elif UI_input.P_amb > P_e_sup:
            # Overexpanded nozzle with shockwaves in exhaust
            result_display = "Overexpanded exhaust with shockwaves in exhaust"
            logger.info(result_display)
            for index in range(len(x_div)):
                A_x = math.pi * (y_div[index] ** 2)
                shift = index + len(x_conv)
                M_x_sup = root_scalar(
                    AT.RS_Area_Mach_X_Y,
                    bracket=[1, 100],
                    args=(UI_input.M_star, A_x, A_star, UI_input.k),
                ).root
                P_x = AT.calc_isen_press(M_x_sup, UI_input.P_0, UI_input.k)
                T_x = AT.calc_isen_temp(M_x_sup, UI_input.T_0, UI_input.k)
                P_array[shift] = P_x
                M_array[shift] = M_x_sup
                T_array[shift] = T_x

            beta, theta = fsolve(
                AT.FS_oblique_angle,
                [np.radians(45), np.radians(45)],
                args=(M_array[-1], P_array[-1], UI_input.P_amb, UI_input.k),
            )
            x_over_shock_1 = x_div[-1]
            y_over_shock_1 = y_div[-1]
            hyp = y_over_shock_1 * 0.6
            x_over_shock_2 = x_over_shock_1 + hyp * np.cos(beta)
            y_over_shock_2 = y_over_shock_1 - hyp * np.sin(beta)
            x_over_shock = [x_over_shock_1, x_over_shock_2]
            y_over_shock = [y_over_shock_1, y_over_shock_2]

        else:
            # Underexpanded nozzle
            result_display = "Underexpanded supersonic exhaust"
            logger.info(result_display)
            # Divergent section - Supersonic flow
            for index in range(len(x_div)):
                A_x = math.pi * (y_div[index] ** 2)
                shift = index + len(x_conv)
                
                M_x_sup = root_scalar(
                    AT.RS_Area_Mach_X_Y,
                    bracket=[1, 100],
                    args=(UI_input.M_star, A_x, A_star, UI_input.k),
                ).root
                P_x = AT.calc_isen_press(M_x_sup, UI_input.P_0, UI_input.k)
                T_x = AT.calc_isen_temp(M_x_sup, UI_input.T_0, UI_input.k)
                P_array[shift] = P_x
                M_array[shift] = M_x_sup
                T_array[shift] = T_x

            # Prandtl-Meyer Expansion Fan
            nu_1 = AT.calc_prandtl_meyer(M_array[-1], UI_input.k)
            M_2 = root_scalar(
                AT.RS_Mach_Press_Isen,
                bracket=[1, 100],
                args=(
                    M_array[-1],
                    UI_input.P_amb + 0.1,
                    P_array[-1],
                    UI_input.k,
                ),
            ).root
            nu_2 = AT.calc_prandtl_meyer(M_2, UI_input.k)
            mu_1 = np.arcsin(1 / M_array[-1])
            mu_2 = np.arcsin(1 / M_2)
            theta = nu_2 - nu_1

            # Plotting fan by sweeping from mu_1 to mu_2
            x_under_shock_1 = x_div[-1]
            y_under_shock_1 = y_div[-1]
            hyp = y_under_shock_1 * 0.6
            theta_nu_2 = theta - mu_2
            theta_range = np.linspace(mu_1, theta_nu_2, 3)
            x_under_shock_2 = np.zeros(len(theta_range))
            y_under_shock_2 = np.zeros(len(theta_range))
            for i in range(len(theta_range)):
                x_under_shock_2[i] = (
                    x_under_shock_1 + np.cos(theta_range[i]) * hyp
                )
                y_under_shock_2[i] = (
                    y_under_shock_1 - np.sin(theta_range[i]) * hyp
                )

        x_under_shock = [x_under_shock_1, x_under_shock_2]
        y_under_shock = [y_under_shock_1, y_under_shock_2]
        x_over_shock = [x_over_shock_1, x_over_shock_2]
        y_over_shock = [y_over_shock_1, y_over_shock_2]

        # Exit area calculations
        P_e = P_array[-1]
        M_e = M_array[-1]
        T_e = T_array[-1]
        rho_e = P_e / (UI_input.R_spec * T_e)
        c_e = (UI_input.k * UI_input.R_spec * T_e) ** 0.5
        V_e = M_e * c_e
        m_dot = rho_e * A_outlet * M_e * c_e
        thr = m_dot * V_e + (P_e - UI_input.P_amb) * A_outlet
        ISP = thr / (m_dot * 9.81)
        expansion_ratio = A_outlet / A_star
        UI_input.rho_0 = PropsSI(
            "D", "P", UI_input.P_0, "T", UI_input.T_0, UI_input.fluid
        )
        V_0 = UI_input.len_inlet * (np.pi * (UI_input.r_inlet**2))
        m_prop = UI_input.rho_0 * V_0

        # Convergent section
        logger.info("Calculating convergent section...")
        for index in range(len(x_conv)):
            A_x = math.pi * (y_conv[index] ** 2)
            M_x_sub = root_scalar(
                AT.RS_Area_Mach_X_Y,
                bracket=[0.0001, 1],
                args=(UI_input.M_star, A_x, A_star, UI_input.k),
            ).root
            P_x = AT.calc_isen_press(M_x_sub, UI_input.P_0, UI_input.k)
            T_x = UI_input.T_0 * (
                (1 + ((UI_input.k - 1) / 2) * M_x_sub * M_x_sub) ** -1
            )

            P_array[index] = P_x
            M_array[index] = M_x_sub
            T_array[index] = T_x

        flow_result = FlowResults(
            x_conv,
            y_conv,
            x_div,
            y_div,
            P_array,
            M_array,
            T_array,
            P_e_sup,
            P_e_sub,
            P_e_shock,
            P_star_shock,
            m_dot,
            P_e,
            expansion_ratio,
            m_prop,
            ISP,
            thr,
            result_display,
            x_under_shock,
            y_under_shock,
            x_over_shock,
            y_over_shock,
        )

        return UI_input, flow_result

    def calc_opt_geom(self, UI_input, flow_result, max_iterations=1000):
        """
        Calculates the optimal nozzle geometry using a gradient descent
        algorithm with an ADAM optimizer to match design thrust.

        Inputs:
            UI_input (UI_Input) - User interface input parameters
            flow_result (FlowResults) - Flow results from calc_thermo()
            max_iterations (int) - Maximum number of iterations
            learning_rate (float) - Learning rate for optimizer
            progress_callback (None) - Marker
        """

        def calc_cost(UI_input):
            """
            Calculates the cost function to be minimized

            Returns:
                (float) - Cost function
            """
            result = self.calc_thermo(UI_input)
            flow_result = result[1]
            thr_curr = flow_result.thr
            return ((thr_curr - UI_input.thr_design), flow_result)

        def calc_gradient(UI_input):
            """
            Calculates the gradient of thrust with respect to throat radius
            dT/dUI_input.r_throat.

            Inputs:
                delta (float) - Small increment to throat area for numerical
                differentiation

            Returns:
                (float) - Gradient of thrust with respect to throat radius
            """
            area_throat = np.pi * (UI_input.r_throat**2)
            delta = area_throat * 0.01
            area_throat += delta
            UI_input.r_throat = (area_throat / np.pi) ** 0.5
            area_outlet = area_throat * area_ratio_outlet
            UI_input.r_outlet = (area_outlet / np.pi) ** 0.5
            area_inlet = area_throat * area_ratio_inlet
            UI_input.r_inlet = (area_inlet / np.pi) ** 0.5
            cost_plus, _ = calc_cost(UI_input)

            area_throat -= 2 * delta
            UI_input.r_throat = (area_throat / np.pi) ** 0.5
            area_outlet = area_throat * area_ratio_outlet
            UI_input.r_outlet = (area_outlet / np.pi) ** 0.5
            area_inlet = area_throat * area_ratio_inlet
            UI_input.r_inlet = (area_inlet / np.pi) ** 0.5
            cost_minus, _ = calc_cost(UI_input)

            # Return UI_input to original
            area_throat += delta
            UI_input.r_throat = (area_throat / np.pi) ** 0.5
            area_outlet = area_throat * area_ratio_outlet
            UI_input.r_outlet = (area_outlet / np.pi) ** 0.5
            area_inlet = area_throat * area_ratio_inlet
            UI_input.r_inlet = (area_inlet / np.pi) ** 0.5
            cost_plus, _ = calc_cost(UI_input)

            return (cost_plus - cost_minus) / (2 * delta)

        logger.info("Calculating optimal nozzle geometry...")
        area_throat = math.pi * (UI_input.r_throat**2)
        area_inlet = math.pi * (UI_input.r_inlet**2)
        area_outlet = math.pi * (UI_input.r_outlet**2)
        area_ratio_outlet = area_outlet / area_throat
        area_ratio_inlet = area_inlet / area_throat
        tol = UI_input.thr_design / 300  # Tolerance for convergence
        dC_dr = calc_gradient(UI_input)
        cost_curr = calc_cost(UI_input)[0]
        learning_rate = (1 / dC_dr) * cost_curr * 0.1
        optimizer = ADAM_Optimizer(learning_rate)

        for iteration in range(max_iterations):
            time.sleep(0.25)
            cost_curr, flow_result = calc_cost(UI_input)

            if abs(cost_curr) < tol:
                logger.info(
                    f"Convergence achieved after {iteration:.0f} iterations"
                )
                return UI_input, flow_result

            gradient = calc_gradient(UI_input)
            learn_rate = (1 / dC_dr) * cost_curr * 0.1
            update = optimizer.update(gradient, learn_rate)

            if iteration % 4 == 0:
                logger.info(
                    f"Iteration {iteration:.0f} of {max_iterations:.0f}"
                )
                yield UI_input, flow_result, int(
                    100 * iteration / max_iterations
                )

            area_throat -= update
            UI_input.r_throat = (area_throat / np.pi) ** 0.5
            area_outlet = area_throat * area_ratio_outlet
            UI_input.r_outlet = (area_outlet / np.pi) ** 0.5
            area_inlet = area_throat * area_ratio_inlet
            UI_input.r_inlet = (area_inlet / np.pi) ** 0.5

        else:
            logger.error(
                f"Convergence failed after {max_iterations:.0f} iterations"
            )

    def calc_depress(self, UI_input, flow_result, max_iterations=100):
        """
        Calculates flow properties during isothermal or adiabatic
        depressurization. Assumes choked flow and uses derivative eqns. with
        Euler's method.
        """
        logger.info("Calculating depressurization conditions...")
        # Initial Conditions
        t = 0
        i = 0
        UI_input.P_0_init = UI_input.P_0
        UI_input.T_0_init = UI_input.T_0
        UI_input.rho_0_init = PropsSI(
            "D", "P", UI_input.P_0_init, "T", UI_input.T_0_init, UI_input.fluid
        )
        V_init = UI_input.len_inlet * (np.pi * (UI_input.r_inlet**2))
        m_init = UI_input.rho_0_init * V_init
        P_curr = UI_input.P_0
        T_curr = UI_input.T_0
        rho_curr = PropsSI("D", "P", P_curr, "T", T_curr, UI_input.fluid)
        m_curr = V_init * rho_curr
        A_star = math.pi * (UI_input.r_throat**2)
        C_d = 1

        P_depress_array = []
        m_depress_array = []
        thr_depress_array = []
        temp_depress_array = []

        AS = AbstractState("HEOS", UI_input.fluid)

        c_0 = np.sqrt(UI_input.k * UI_input.R_spec * UI_input.T_0_init)
        exp = (UI_input.k + 1) / (2 * UI_input.k - 2)
        tau = (V_init / (C_d * A_star * c_0)) * (((UI_input.k + 1) / 2) ** exp)
        decay_time = 4.61 * tau  # Time until 1 percent remaining
        time_step = decay_time / max_iterations

        if UI_input.depress_type == "Isothermal":
            calc_dPdt = lambda P, tau, t: AT.calc_isotherm_dPdt(P, tau, t)
            calc_drhodt = lambda rho, tau, t: AT.calc_isotherm_drhodt(
                rho, tau, t
            )
            calc_dTdt = lambda T, tau, t: 0
        elif UI_input.depress_type == "Adiabatic":
            calc_dPdt = lambda P, tau, t: AT.calc_ada_dPdT(
                P, tau, t, UI_input.k
            )
            calc_drhodt = lambda rho, tau, t: AT.calc_ada_drhodt(
                rho, tau, t, UI_input.k
            )
            calc_dTdt = lambda T, tau, t: AT.calc_ada_dTdt(
                T, tau, t, UI_input.k
            )

        # Main depressurization loop
        while m_curr > m_init * 0.01 and t < decay_time:
            time.sleep(0.25)
            t += time_step
            i += 1
            dPdt = calc_dPdt(UI_input.P_0_init, tau, t)
            drhodt = calc_drhodt(UI_input.rho_0_init, tau, t)
            dTdt = calc_dTdt(UI_input.T_0_init, tau, t)

            dP = dPdt * time_step
            drho = drhodt * time_step
            dT = dTdt * time_step

            P_curr += dP
            rho_curr += drho
            T_curr += dT
            m_curr = rho_curr * V_init
            P_depress_array.append(P_curr)
            m_depress_array.append(m_curr)
            temp_depress_array.append(T_curr)

            UI_input.P_0 = P_curr
            UI_input, flow_result = self.calc_thermo(UI_input)
            thr_curr = flow_result.thr
            thr_depress_array.append(thr_curr)
            if i % 10 == 0:
                logger.info(f"Iteration: {i:.0f} of {max_iterations:.0f}")
                yield (
                    UI_input,
                    flow_result,
                    t,
                    P_curr,
                    m_curr,
                    thr_curr,
                    T_curr,
                    i / max_iterations,
                )

        P_depress_array = np.array(P_depress_array)
        m_depress_array = np.array(m_depress_array)
        thr_depress_array = np.array(thr_depress_array)
        temp_depress_array = np.array(temp_depress_array)
        t_depress_array = np.linspace(time_step, t, len(P_depress_array))

        depress_result = (
            UI_input,
            flow_result,
            t_depress_array,
            P_depress_array,
            m_depress_array,
            thr_depress_array,
            temp_depress_array,
        )
        logger.info("Depressurization complete.")
        return depress_result

    def calc_monte_carlo(self, UI_input, flow_result, N=50, sigma_frac=0.01):
        """Runs a Monte Carlo simulation using linear hypercube scaling of an
        input parameter to predict thrust variance.

        Assumes input parameter is normally distributed with a standard
        deviation that is a fraction of the mean.

        Inputs:
            UI_input (dataclass) - User inputs
            flow_result (dataclass) - Flow results
            N (int) - Number of samples to generate
            sigma_frac (float) - fraction of mean used as standard deviation
        Returns:
            mc_thrust_array (np.array) - Array of thrust samples
        """

        logger.info("Running Monte Carlo...")
        param_map = {
            "Chamber Pressure": "P_0",
            "Chamber Temperature": "T_0",
            "Ambient Pressure": "P_amb",
            "Throat Radius": "r_throat",
            "Outlet Radius": "r_outlet",
        }

        param = param_map[UI_input.monte_carlo_type]
        mu = getattr(UI_input, param)
        sigma = mu * sigma_frac
        mid = (np.arange(1, N + 1) - 0.5) / N
        input_var_array = norm.ppf(mid, loc=mu, scale=sigma)
        mc_thrust_array = np.zeros(N)

        for idx, var in enumerate(input_var_array):
            # Check if variance doesn't produce negative value
            if var < 0:
                var = 1e-6
            setattr(UI_input, param, var)
            UI_input, flow_result = self.calc_thermo(UI_input)
            mc_thrust_array[idx] = flow_result.thr

        logger.info("Monte Carlo complete")

        return mc_thrust_array
