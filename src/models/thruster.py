import numpy as np
import math
from dataclasses import dataclass
import Aero_Thermo as AT
import Method_Of_Char as MOC
from Optimizer import ADAM_Optimizer
from scipy.optimize import (
    fsolve,
    root_scalar)
from CoolProp.CoolProp import PropsSI, AbstractState

@dataclass
class FlowResults:
    x_conv: np.ndarray
    y_conv: np.ndarray
    x_div: np.ndarray
    y_div:  np.ndarray
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

class ThrusterModel():
    def calc_thermo(self, UI_input):
        A_star = math.pi*(UI_input.r_throat**2)
        A_inlet = math.pi*(UI_input.r_inlet**2)
        A_outlet = math.pi*(UI_input.r_outlet**2)
        res = 150
        converg_length = (UI_input.r_inlet-UI_input.r_throat)/np.tan(
            UI_input.converg_angle)
        diverg_length = (UI_input.r_outlet-UI_input.r_throat)/np.tan(
            UI_input.diverg_angle)
        x_conv = -np.flip(np.linspace(0, converg_length, res))
        y_conv = UI_input.r_throat -x_conv*np.tan(UI_input.converg_angle)

        if UI_input.noz_type == "MOC Full Length Nozzle":
            x_div, y_div, *_ = MOC.gen_MOC_FLN(UI_input.M_exit_moc,
                                               UI_input.r_throat,
                                               UI_input.k,
                                               div=50)
            UI_input.r_outlet = np.max(y_div)
            A_outlet = math.pi*(UI_input.r_outlet**2)
        elif UI_input.noz_type == "MOC Minimum Length Nozzle":
            x_div, y_div, *_ = MOC.gen_MOC_MLN(UI_input.M_exit_moc,
                                               UI_input.r_throat,
                                               UI_input.k,
                                               div=50)
            UI_input.r_outlet = np.max(y_div)
            A_outlet = math.pi*(UI_input.r_outlet**2)
        elif UI_input.noz_type == "Conical":
            x_div = np.linspace(0, diverg_length, res)
            y_div = UI_input.r_throat + x_div*np.tan(UI_input.diverg_angle)

        grey_out_style = """QLineEdit[readOnly="true"]
            {Background-color: #a3a3a3; color: white;}
            QLineEdit[readOnly="false"]
            { Background-color: white; color: blacUI_input.k;}"""
        # converg_ang_val.setStyleSheet(grey_out_style)
        # UI_input.diverg_angle_val.setStyleSheet(grey_out_style)
        # radius_exit_val.setStyleSheet(grey_out_style)
        # UI_input.M_exit_moc_val.setStyleSheet(grey_out_style)

        try:
            M_e_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                    args=(UI_input.M_star, A_outlet,
                                        A_star, UI_input.k)).root
            M_e_sub = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                    args=(UI_input.M_star, A_outlet,
                                        A_star, UI_input.k)).root
        except ValueError as e:
            print("Unable to solve for Mach numbers." +
                    "Expand solver bracket to ensure solution exists.")

        # Preallocation
        P_array = np.zeros(len(x_conv)+len(x_div))
        M_array = np.zeros(len(x_conv)+len(x_div))
        T_array = np.zeros(len(x_conv)+len(x_div))

        P_e_sup = AT.calc_isen_press(M_e_sup,UI_input.P_0,UI_input.k)
        P_e_sub = AT.calc_isen_press(M_e_sub,UI_input.P_0,UI_input.k)

        x_shock = None      # Shock location
        x_under_shock_1, y_under_shock_1 = None, None
        x_under_shock_2, y_under_shock_2 = None, None
        x_over_shock_1, y_over_shock_1 = None, None
        x_over_shock_2, y_over_shock_2 = None, None

        # Solve for shock at exit of nozzle
        A_x = math.pi*(y_div[-1]**2)
        # Mach number before shock
        M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                args=(UI_input.M_star, A_x, A_star,
                                    UI_input.k)).root
        P_x = AT.calc_isen_press(M_x_sup,UI_input.P_0,UI_input.k)
        T_x = AT.calc_isen_temp(M_x_sup,UI_input.T_0,UI_input.k)
        M_y,P_y = AT.calc_M_P_normal(M_x_sup,P_x,UI_input.k)
        UI_input.P_0_y = AT.calc_isen_stag_press(M_y,P_y,UI_input.k)
        UI_input.T_0_y = UI_input.T_0

        # New critical area using post shocUI_input.k conditions
        A_star_shock = A_x * M_y * (
            ((2/(UI_input.k+1))*(1+((UI_input.k-1)/2)*M_y*M_y))**(
                (-UI_input.k-1)/(2*UI_input.k-2)))
        M_y_e = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.00001,1],
                            args=(UI_input.M_star,A_outlet,
                                    A_star_shock,UI_input.k)).root
        P_e_shock = AT.calc_isen_press(M_y_e,UI_input.P_0_y,UI_input.k)

        # Solve for shock at onset of divergent section
        A_x = math.pi*(y_div[1]**2)
        M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                args=(UI_input.M_star, A_x, A_star,
                                    UI_input.k)).root
        P_x = AT.calc_isen_press(M_x_sup,UI_input.P_0,UI_input.k)
        T_x = AT.calc_isen_temp(M_x_sup,UI_input.T_0,UI_input.k)
        M_y,P_y = AT.calc_M_P_normal(M_x_sup,P_x,UI_input.k)
        UI_input.P_0_y = AT.calc_isen_stag_press(M_y,P_y,UI_input.k)
        UI_input.T_0_y = UI_input.T_0

        A_star_shock = A_x * M_y * (
            ((2/(UI_input.k+1))*(1+((UI_input.k-1)/2)*M_y*M_y))**(
                (-UI_input.k-1)/(2*UI_input.k-2)))
        M_y_e = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                            args=(UI_input.M_star,A_outlet,
                                    A_star_shock,UI_input.k)).root
        P_star_shock = AT.calc_isen_press(M_y_e,UI_input.P_0_y,UI_input.k)

        if abs(UI_input.P_amb-P_e_sup)/P_e_sup < 0.1 or (
            UI_input.P_amb < 50 and P_e_sup < 100):
            # Check if Back pressure and supersonic exit pressure are close
            # enough for perfect expansion
            result_display= 'Perfectly expanded supersonic exhaust!'
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]**2)
                shift = index + len(x_conv)
                M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                        args=(UI_input.M_star,A_x,
                                            A_star,UI_input.k)).root
                P_x = AT.calc_isen_press(M_x_sup,UI_input.P_0,UI_input.k)
                T_x = AT.calc_isen_temp(M_x_sup,UI_input.T_0,UI_input.k)
                P_array[shift] = P_x
                M_array[shift] = M_x_sup
                T_array[shift] = T_x

        elif UI_input.P_amb >= UI_input.P_0:
            result_display = "Back pressure high enough to generate reversed flow"
        elif UI_input.P_amb >= P_e_sub:
            # Back pressure is too high for choked subsonic flow
            result_display = "Back pressure too high for choked subsonic flow"
            M_e_sub = root_scalar(
                AT.RS_Mach_Press_Isen,
                bracket=[0.0001,1],
                args=(0,UI_input.P_amb,UI_input.P_0,UI_input.k)).root
            UI_input.M_star = root_scalar(AT.RS_Area_Mach_X_Y,
                                        bracket=[0.0001,1], 
                                        args=(M_e_sub, A_star,
                                                A_outlet, UI_input.k)).root
            
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]**2)
                shift = index + len(x_conv)

                M_x = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                    args=(M_e_sub, A_x, A_outlet, 
                                        UI_input.k)).root
                P_x = AT.calc_isen_press(M_x,UI_input.P_0,UI_input.k)
                T_x = AT.calc_isen_temp(M_x,UI_input.T_0,UI_input.k)

                P_array[shift] = P_x
                M_array[shift] = M_x
                T_array[shift] = T_x

        elif UI_input.P_amb > P_e_shock:
            """Back pressure is low enough for choked supersonic flow with
            possible normal shock"""
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]**2)
                shift = index + len(x_conv)
                if x_shock is None:
                    # Pre shock wave prop
                    M_x_sup = root_scalar(
                        AT.RS_Area_Mach_X_Y,
                        bracket=[1,100],
                        args=(UI_input.M_star, A_x, A_star, UI_input.k)
                        ).root
                    P_x = AT.calc_isen_press(M_x_sup,UI_input.P_0,UI_input.k)
                    T_x = AT.calc_isen_temp(M_x_sup,UI_input.T_0,UI_input.k)
                    P_array[shift] = P_x
                    M_array[shift] = M_x_sup
                    T_array[shift] = T_x

                    # Post shock wave prop
                    M_y, P_y = AT.calc_M_P_normal(M_x_sup,P_x,UI_input.k)
                    UI_input.P_0_y = AT.calc_isen_stag_press(M_y,P_y,UI_input.k)
                    UI_input.T_0_y = UI_input.T_0

                    A_star_shock = A_x * M_y * (
                        ((2/(UI_input.k+1))*(1+((UI_input.k-1)/2)*M_y*M_y))**(
                            (-UI_input.k-1)/(2*UI_input.k-2)))
                    M_y_e = root_scalar(
                        AT.RS_Area_Mach_X_Y,
                        bracket=[0.0001,1], 
                        args=(UI_input.M_star, A_outlet,
                                A_star_shock, UI_input.k)).root   
                    P_y_e = AT.calc_isen_press(M_y_e, UI_input.P_0_y, UI_input.k)
                    T_y_e = UI_input.T_0_y * ((1+((UI_input.k-1)/2)*M_y_e*M_y_e)
                                            **-1)

                    if abs((P_y_e - UI_input.P_amb)/P_y_e) < 0.1:
                        print('Shock location calculated')
                        x_shock = x_div[index]
                        result_display = f"""Normal shock generated in
                                        divergent section at {x_shock:.3f} m"""
                else:
                    M_y_curr = root_scalar(
                        AT.RS_Area_Mach_X_Y,
                        bracket=[0.0001,1],
                        args=(UI_input.M_star, A_x, A_star_shock, 
                                UI_input.k)).root
                    P_y_curr = AT.calc_isen_press(M_y_curr, UI_input.P_0_y, 
                                                    UI_input.k)
                    T_y_curr = AT.calc_isen_temp(M_y_curr, UI_input.T_0_y, 
                                                    UI_input.k)
                    P_array[shift] = P_y_curr
                    M_array[shift] = M_y_curr
                    T_array[shift] = T_y_curr

        elif UI_input.P_amb > P_e_sup:
            # Overexpanded nozzle with shockwaves in exhaust
            result_display = "Overexpanded exhaust with shockwaves in exhaust"
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]**2)
                shift = index + len(x_conv)
                M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                        args=(UI_input.M_star, A_x,
                                            A_star, UI_input.k)).root
                P_x = AT.calc_isen_press(M_x_sup,UI_input.P_0,UI_input.k)
                T_x = AT.calc_isen_temp(M_x_sup,UI_input.T_0,UI_input.k)
                P_array[shift] = P_x
                M_array[shift] = M_x_sup
                T_array[shift] = T_x

            beta, theta = fsolve(
                AT.FS_oblique_angle,
                [np.radians(45),np.radians(45)],
                args=(M_array[-1], P_array[-1], UI_input.P_amb, 
                        UI_input.k))
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
            
            # Divergent section - Supersonic flow
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]*y_div[index])
                shift = index + len(x_conv)
                M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                        args=(UI_input.M_star, A_x,
                                            A_star, UI_input.k)).root
                P_x = AT.calc_isen_press(M_x_sup,UI_input.P_0,UI_input.k)
                T_x = AT.calc_isen_temp(M_x_sup,UI_input.T_0,UI_input.k)
                P_array[shift] = P_x
                M_array[shift] = M_x_sup
                T_array[shift] = T_x


            # Prandtl-Meyer Expansion Fan
            nu_1 = AT.calc_prandtl_meyer(M_array[-1],UI_input.k)
            M_2 = root_scalar(AT.RS_Mach_Press_Isen, bracket=[1,100],
                                args=(M_array[-1],UI_input.P_amb+.1,
                                    P_array[-1],UI_input.k)).root
            nu_2 = AT.calc_prandtl_meyer(M_2,UI_input.k)
            mu_1 = np.arcsin(1/M_array[-1])
            mu_2 = np.arcsin(1/M_2)
            theta = nu_2 - nu_1

            # Plotting fan by sweeping from mu_1 to mu_2
            x_under_shock_1 = x_div[-1]
            y_under_shock_1 = y_div[-1]
            hyp = y_under_shock_1 * .6
            theta_nu_2 = theta - mu_2
            theta_range = np.linspace(mu_1, theta_nu_2,3)
            x_under_shock_2 = np.zeros(len(theta_range))
            y_under_shock_2 = np.zeros(len(theta_range))
            for i in range(len(theta_range)):
                x_under_shock_2[i] = x_under_shock_1 + np.cos(
                    theta_range[i]) * hyp
                y_under_shock_2[i] = y_under_shock_1 - np.sin(
                    theta_range[i]) * hyp
                
        x_under_shock = [x_under_shock_1, x_under_shock_2]
        y_under_shock = [y_under_shock_1, y_under_shock_2]
        x_over_shock = [x_over_shock_1, x_over_shock_2]
        y_over_shock = [y_over_shock_1, y_over_shock_2]

        # Exit area calculations
        P_e=P_array[-1]
        M_e=M_array[-1]
        T_e=T_array[-1]
        rho_e = P_e/(UI_input.R_spec*T_e)
        c_e = (UI_input.k*UI_input.R_spec*T_e)**0.5
        V_e = M_e * c_e
        m_dot = rho_e * A_outlet * M_e * c_e
        thr = m_dot * V_e + (
            P_e - UI_input.P_amb) * A_outlet
        ISP = thr/(m_dot*9.81)
        expansion_ratio = A_outlet/A_star
        UI_input.rho_0 = PropsSI('D', 'P', UI_input.P_0, 'T', UI_input.T_0,
                                 UI_input.fluid)
        V_0 = UI_input.len_inlet * (np.pi * (UI_input.r_inlet**2))
        m_prop = UI_input.rho_0 * V_0 

        # Convergent section
        for index in range(len(x_conv)):
            A_x = math.pi*(y_conv[index]**2)
            M_x_sub = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                    args=(UI_input.M_star, A_x, A_star,
                                        UI_input.k)).root
            P_x = AT.calc_isen_press(M_x_sub,UI_input.P_0,UI_input.k)
            T_x = UI_input.T_0 * ((1+ ((UI_input.k-1)/2)*M_x_sub*M_x_sub)**-1)

            P_array[index] = P_x
            M_array[index] = M_x_sub
            T_array[index] = T_x

        flow_result = FlowResults(x_conv, y_conv, x_div, y_div, P_array,
                                  M_array, T_array, P_e_sup, P_e_sub,
                                  P_e_shock, P_star_shock, m_dot, P_e,
                                  expansion_ratio, m_prop, ISP, thr,
                                  result_display, x_under_shock, y_under_shock,
                                  x_over_shock, y_over_shock)

        return flow_result
    
    def calc_opt_geom(self, UI_input, flow_result, max_iterations=1000,
                      learning_rate=1E-3):
        """
        Calculates the optimal nozzle geometry using a gradient descent
        algorithm with an ADAM optimizer to match design thrust.
        """

        A_star = math.pi*(UI_input.r_throat**2)
        A_inlet = math.pi*(UI_input.r_inlet**2)
        A_outlet = math.pi*(UI_input.r_outlet**2)
        area_ratio_outlet = A_outlet / A_star
        area_ratio_inlet = A_inlet / A_star
        tol=UI_input.thr_design/10000            # Tolerance for convergence
        optimizer = ADAM_Optimizer(learning_rate)

        def calc_cost(UI_input):
            """
            Calculates the cost function to be minimized

            Returns:
                (float) - Cost function
            """
            flow_result = self.calc_thermo(UI_input)
            thr_curr = flow_result.thr
            return (thr_curr - UI_input.thr_design)**2

        def calc_gradient(UI_input, delta=.0001):
            """
            Calculates the gradient of thrust with respect to throat radius 
            dT/dUI_input.r_throat.

            Inputs:
                delta (float) - Small increment to throat radius for numerical
                differentiation
            
            Returns:
                (float) - Gradient of thrust with respect to throat radius
            """
            UI_input.r_throat += delta
            UI_input.r_outlet = ((UI_input.r_throat**2) * area_ratio_outlet)**0.5
            UI_input.r_inlet = ((UI_input.r_throat**2) * area_ratio_inlet)**0.5
            
            cost_plus = calc_cost(UI_input)
            UI_input.r_throat -= delta
            UI_input.r_outlet = ((UI_input.r_throat**2) * area_ratio_outlet)**0.5
            UI_input.r_inlet = ((UI_input.r_throat**2) * area_ratio_inlet)**0.5
            cost_minus = calc_cost(UI_input)
            return (cost_plus - cost_minus) / (2*delta)            

        for iteration in range(max_iterations):
            print(iteration)
            cost_curr = calc_cost(UI_input)
            print(cost_curr)

            if abs(cost_curr) < tol:
                break

            gradient = calc_gradient(UI_input)
            print(gradient)
            update = optimizer.update(gradient)
            print(update)

            UI_input.r_throat -= update
            UI_input.r_outlet = ((UI_input.r_throat**2) * area_ratio_outlet)**0.5
            UI_input.r_inlet = ((UI_input.r_throat**2) * area_ratio_inlet)**0.5

        else:
            print(f"Convergence failed after {max_iterations:.0f} iterations")


    def calc_depress(self, UI_input, flow_result):
        """
        Calculates flow properties during isothermal or adiabatic
        depressurization. Assumes choked flow and uses derivative eqns. with
        Euler's method.
        """

        # Initial Conditions
        time_step = .00001
        t = 0
        UI_input.P_0_init = UI_input.P_0
        UI_input.T_0_init = UI_input.T_0
        UI_input.rho_0_init = PropsSI('D', 'P', UI_input.P_0_init, 'T',
                                      UI_input.T_0_init, UI_input.fluid)
        V_init = UI_input.len_inlet * (np.pi * (UI_input.r_inlet**2))
        m_init = UI_input.rho_0_init * V_init
        P_curr = UI_input.P_0
        T_curr = UI_input.T_0
        rho_curr = PropsSI('D', 'P', P_curr, 'T', T_curr, UI_input.fluid)
        m_curr = V_init * rho_curr
        C_d = 1
        
        P_depress_array = []
        m_depress_array = []
        thr_depress_array = []
        temp_depress_array = []

        AS = AbstractState("HEOS", UI_input.fluid)

        c_0 = np.sqrt(UI_input.k*UI_input.R_spec * UI_input.T_0_init)
        exp = (UI_input.k + 1) / (2 * UI_input.k - 2)
        tau = (V_init / (C_d * A_star * c_0))*(((UI_input.k+1)/2)**exp)
        decay_time = 4.61 * tau         # Time until 1 percent remaining
        if decay_time/1000 > time_step: # Check if timestep too small
            time_step = decay_time/1000

        if depress_type_list.currentIndex() == 0:
            calc_dPdt = lambda P, tau, t: AT.calc_isotherm_dPdt(P, tau, t)
            calc_drhodt = lambda rho, tau, t: AT.calc_isotherm_drhodt(
                rho, tau, t)
            calc_dTdt = lambda T, tau, t: 0
        elif depress_type_list.currentIndex() == 1:
            calc_dPdt = lambda P, tau, t: AT.calc_ada_dPdT(P, tau, t, UI_input.k)
            calc_drhodt = lambda rho, tau, t: AT.calc_ada_drhodt(
                rho, tau, t, UI_input.k)
            calc_dTdt = lambda T, tau, t: AT.calc_ada_dTdt(T, tau, t, UI_input.k)

        # Main depressurization loop
        while m_curr > m_init*0.01 and t < decay_time:
            t += time_step
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
            calc_thermo()
            thr_depress_array.append(thr)
            QApplication.processEvents()

        P_depress_array = np.array(P_depress_array)
        m_depress_array = np.array(m_depress_array)
        thr_depress_array = np.array(thr_depress_array)
        t_depress_array = np.linspace(time_step, t, len(P_depress_array))

        canvas.axes_depress.plot(t_depress_array, P_depress_array, 'g-')
        canvas.axes_mass.plot(t_depress_array, m_depress_array, 'r-')
        canvas.axes_thrust.plot(t_depress_array, thr_depress_array, 'b-')   
        canvas.axes_detemp.plot(t_depress_array, temp_depress_array, 'w-')

        depress_button.setText("Calculation Complete")
        QApplication.processEvents()
        canvas.draw()

    