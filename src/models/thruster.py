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

class ThrusterModel():
    def calc_thermo(self, UIInputs):
        fluid = UIInputs.fluid
        R_spec = UIInputs.R_spec
        k = UIInputs.k
        T_0 = UIInputs.T_0
        P_0 = UIInputs.P_0
        rho_0 = UIInputs.rho_0
        P_amb = UIInputs.P_amb
        T_star = UIInputs.T_star
        P_star = UIInputs.P_star
        M_star = UIInputs.M_star
        r_throat = UIInputs.r_throat
        r_inlet = UIInputs.r_inlet
        r_outlet = UIInputs.r_outlet
        converg_angle = UIInputs.converg_angle
        diverg_angle = UIInputs.diverg_angle
        len_inlet = UIInputs.len_inlet
        M_exit = UIInputs.M_exit
        thr_design = UIInputs.thr_design
        noz_type = UIInputs.noz_type

        A_star = math.pi*(r_throat**2)
        A_inlet = math.pi*(r_inlet**2)
        A_outlet = math.pi*(r_outlet**2)
        res = 150
        converg_length = (r_inlet-r_throat)/np.tan(
            converg_angle)
        diverg_length = (r_outlet-r_throat)/np.tan(
            diverg_angle)
        x_conv = -np.flip(np.linspace(0, converg_length, res))
        y_conv = r_throat -x_conv*np.tan(converg_angle)

        if noz_type == "MOC Full Length Nozzle":
            x_div, y_div, *_ = MOC.gen_MOC_FLN(M_exit, r_throat,
                                                            k=k, div=50)
            r_outlet = np.max(y_div)
            A_outlet = math.pi*(r_outlet**2)
        elif noz_type == "MOC Minimum Length Nozzle":
            x_div, y_div, *_ = MOC.gen_MOC_MLN(M_exit, r_throat,
                                                            k=k, div=50)
            r_outlet = np.max(y_div)
            A_outlet = math.pi*(r_outlet**2)
        elif noz_type == "Conical":
            x_div = np.linspace(0, diverg_length, res)
            y_div = r_throat + x_div*np.tan(diverg_angle)

        grey_out_style = """QLineEdit[readOnly="true"]
            {background-color: #a3a3a3; color: white;}
            QLineEdit[readOnly="false"]
            { background-color: white; color: black;}"""
        # converg_ang_val.setStyleSheet(grey_out_style)
        # diverg_angle_val.setStyleSheet(grey_out_style)
        # radius_exit_val.setStyleSheet(grey_out_style)
        # M_exit_val.setStyleSheet(grey_out_style)

        try:
            M_e_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                    args=(M_star, A_outlet,
                                        A_star, k)).root
            M_e_sub = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                    args=(M_star, A_outlet,
                                        A_star, k)).root
        except ValueError as e:
            print("Unable to solve for Mach numbers." +
                    "Expand solver bracket to ensure solution exists.")

        # Preallocation
        P_array = np.zeros(len(x_conv)+len(x_div))
        M_array = np.zeros(len(x_conv)+len(x_div))
        T_array = np.zeros(len(x_conv)+len(x_div))

        P_e_sup = AT.calc_isen_press(M_e_sup,P_0,k)
        P_e_sub = AT.calc_isen_press(M_e_sub,P_0,k)

        shock_flag = False  # Checks if shock was calculated
        x_shock = None # Shock location

        # Solve for shock at exit of nozzle
        A_x = math.pi*(y_div[-1]**2)
        # Mach number before shock
        M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                args=(M_star, A_x, A_star,
                                    k)).root
        P_x = AT.calc_isen_press(M_x_sup,P_0,k)
        T_x = AT.calc_isen_temp(M_x_sup,T_0,k)
        M_y,P_y = AT.calc_M_P_normal(M_x_sup,P_x,k)
        P_0_y = AT.calc_isen_stag_press(M_y,P_y,k)
        T_0_y = T_0

        # New critical area using post shock conditions
        A_star_shock = A_x * M_y * (
            ((2/(k+1))*(1+((k-1)/2)*M_y*M_y))**(
                (-k-1)/(2*k-2)))
        M_y_e = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.00001,1],
                            args=(M_star,A_outlet,
                                    A_star_shock,k)).root
        P_e_shock = AT.calc_isen_press(M_y_e,P_0_y,k)

        # Solve for shock at onset of divergent section
        A_x = math.pi*(y_div[1]**2)
        M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                args=(M_star, A_x, A_star,
                                    k)).root
        P_x = AT.calc_isen_press(M_x_sup,P_0,k)
        T_x = AT.calc_isen_temp(M_x_sup,T_0,k)
        M_y,P_y = AT.calc_M_P_normal(M_x_sup,P_x,k)
        P_0_y = AT.calc_isen_stag_press(M_y,P_y,k)
        T_0_y = T_0

        A_star_shock = A_x * M_y * (
            ((2/(k+1))*(1+((k-1)/2)*M_y*M_y))**(
                (-k-1)/(2*k-2)))
        M_y_e = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                            args=(M_star,A_outlet,
                                    A_star_shock,k)).root
        P_star_shock = AT.calc_isen_press(M_y_e,P_0_y,k)

        if abs(P_amb-P_e_sup)/P_e_sup < 0.1 or (
            P_amb < 50 and P_e_sup < 100):
            # Check if back pressure and supersonic exit pressure are close
            # enough for perfect expansion
            result_display= 'Perfectly expanded supersonic exhaust!'
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]**2)
                shift = index + len(x_conv)
                M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                        args=(M_star,A_x,
                                            A_star,k)).root
                P_x = AT.calc_isen_press(M_x_sup,P_0,k)
                T_x = AT.calc_isen_temp(M_x_sup,T_0,k)
                P_array[shift] = P_x
                M_array[shift] = M_x_sup
                T_array[shift] = T_x

        elif P_amb >= P_0:
            result_display = "Back pressure high enough to generate reversed flow"
        elif P_amb >= P_e_sub:
            # Back pressure is too high for choked subsonic flow
            result_display = "Back pressure too high for choked subsonic flow"
            M_e_sub = root_scalar(
                AT.RS_Mach_Press_Isen,
                bracket=[0.0001,1],
                args=(0,P_amb,P_0,k)).root
            M_star = root_scalar(AT.RS_Area_Mach_X_Y,
                                        bracket=[0.0001,1], 
                                        args=(M_e_sub, A_star,
                                                A_outlet, k)).root
            
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]**2)
                shift = index + len(x_conv)

                M_x = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                    args=(M_e_sub, A_x, A_outlet, 
                                        k)).root
                P_x = AT.calc_isen_press(M_x,P_0,k)
                T_x = AT.calc_isen_temp(M_x,T_0,k)

                P_array[shift] = P_x
                M_array[shift] = M_x
                T_array[shift] = T_x

        elif P_amb > P_e_shock:
            """Back pressure is low enough for choked supersonic flow with
            possible normal shock"""
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]**2)
                shift = index + len(x_conv)
                if not shock_flag:
                    # Pre shock wave prop
                    M_x_sup = root_scalar(
                        AT.RS_Area_Mach_X_Y,
                        bracket=[1,100],
                        args=(M_star, A_x, A_star, k)
                        ).root
                    P_x = AT.calc_isen_press(M_x_sup,P_0,k)
                    T_x = AT.calc_isen_temp(M_x_sup,T_0,k)
                    P_array[shift] = P_x
                    M_array[shift] = M_x_sup
                    T_array[shift] = T_x

                    # Post shock wave prop
                    M_y, P_y = AT.calc_M_P_normal(M_x_sup,P_x,k)
                    P_0_y = AT.calc_isen_stag_press(M_y,P_y,k)
                    T_0_y = T_0

                    A_star_shock = A_x * M_y * (
                        ((2/(k+1))*(1+((k-1)/2)*M_y*M_y))**(
                            (-k-1)/(2*k-2)))
                    M_y_e = root_scalar(
                        AT.RS_Area_Mach_X_Y,
                        bracket=[0.0001,1], 
                        args=(M_star, A_outlet,
                                A_star_shock, k)).root   
                    P_y_e = AT.calc_isen_press(M_y_e, P_0_y, k)
                    T_y_e = T_0_y * ((1+((k-1)/2)*M_y_e*M_y_e)
                                            **-1)

                    if abs((P_y_e - P_amb)/P_y_e) < 0.1:
                        print('Shock location calculated')
                        shock_flag = True
                        x_shock = x_div[index]
                        result_display = f"""Normal shock generated in
                                        divergent section at {x_shock:.3f} m"""
                elif shock_flag:
                    M_y_curr = root_scalar(
                        AT.RS_Area_Mach_X_Y,
                        bracket=[0.0001,1],
                        args=(M_star, A_x, A_star_shock, 
                                k)).root
                    P_y_curr = AT.calc_isen_press(M_y_curr, P_0_y, 
                                                    k)
                    T_y_curr = AT.calc_isen_temp(M_y_curr, T_0_y, 
                                                    k)
                    P_array[shift] = P_y_curr
                    M_array[shift] = M_y_curr
                    T_array[shift] = T_y_curr

        elif P_amb > P_e_sup:
            # Overexpanded nozzle with shockwaves in exhaust
            result_display = "Overexpanded exhaust with shockwaves in exhaust"
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]**2)
                shift = index + len(x_conv)
                M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                        args=(M_star, A_x,
                                            A_star, k)).root
                P_x = AT.calc_isen_press(M_x_sup,P_0,k)
                T_x = AT.calc_isen_temp(M_x_sup,T_0,k)
                P_array[shift] = P_x
                M_array[shift] = M_x_sup
                T_array[shift] = T_x

            beta, theta = fsolve(
                AT.FS_oblique_angle,
                [np.radians(45),np.radians(45)],
                args=(M_array[-1], P_array[-1], P_amb, 
                        k))
            x_over_shock_1 = x_div[-1]
            y_over_shock_1 = y_div[-1]
            hyp = y_over_shock_1 * 0.6
            x_over_shock_2 = x_over_shock_1 + hyp * np.cos(beta)
            y_over_shock_2 = y_over_shock_1 - hyp * np.sin(beta)
            # canvas.axes.plot([x_over_shock_1,x_over_shock_2],
            #                         [y_over_shock_1,y_over_shock_2], 'r-')

        else:
            # Underexpanded nozzle
            result_display = "Underexpanded supersonic exhaust"
            
            # Divergent section - Supersonic flow
            for index in range(len(x_div)):
                A_x = math.pi*(y_div[index]*y_div[index])
                shift = index + len(x_conv)
                M_x_sup = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[1,100],
                                        args=(M_star, A_x,
                                            A_star, k)).root
                P_x = AT.calc_isen_press(M_x_sup,P_0,k)
                T_x = AT.calc_isen_temp(M_x_sup,T_0,k)
                P_array[shift] = P_x
                M_array[shift] = M_x_sup
                T_array[shift] = T_x


            # Prandtl-Meyer Expansion Fan
            nu_1 = AT.calc_prandtl_meyer(M_array[-1],k)
            M_2 = root_scalar(AT.RS_Mach_Press_Isen, bracket=[1,100],
                                args=(M_array[-1],P_amb+.1,
                                    P_array[-1],k)).root
            nu_2 = AT.calc_prandtl_meyer(M_2,k)
            mu_1 = np.arcsin(1/M_array[-1])
            mu_2 = np.arcsin(1/M_2)
            theta = nu_2 - nu_1

            # Plotting fan by sweeping from mu_1 to mu_2
            x_under_shock_1 = x_div[-1]
            y_under_shock_1 = y_div[-1]
            hyp = y_under_shock_1 * .6
            theta_nu_2 = theta - mu_2
            theta_range = np.linspace(mu_1, theta_nu_2,3)
            for i in range(len(theta_range)):
                x_under_shock_2 = x_under_shock_1 + np.cos(
                    theta_range[i]) * hyp
                y_under_shock_2 = y_under_shock_1 - np.sin(
                    theta_range[i]) * hyp
                # canvas.axes.plot([x_under_shock_1,x_under_shock_2],
                #                         [y_under_shock_1,y_under_shock_2],
                #                         'g-')

        # Exit area calculations
        P_e=P_array[-1]
        M_e=M_array[-1]
        T_e=T_array[-1]
        rho_e = P_e/(R_spec*T_e)
        c_e = (k*R_spec*T_e)**0.5
        V_e = M_e * c_e
        m_dot = rho_e * A_outlet * M_e * c_e
        thr = m_dot * V_e + (
            P_e - P_amb) * A_outlet
        ISP = thr/(m_dot*9.81)
        expansion_ratio = A_outlet/A_star
        rho_0 = PropsSI('D', 'P', P_0, 'T', T_0, fluid)
        V_0 = len_inlet * (np.pi * (r_inlet**2))
        m_prop = rho_0 * V_0 

        # Convergent section
        for index in range(len(x_conv)):
            A_x = math.pi*(y_conv[index]**2)
            M_x_sub = root_scalar(AT.RS_Area_Mach_X_Y, bracket=[0.0001,1],
                                    args=(M_star, A_x, A_star,
                                        k)).root
            P_x = AT.calc_isen_press(M_x_sub,P_0,k)
            T_x = T_0 * ((1+ ((k-1)/2)*M_x_sub*M_x_sub)**-1)

            P_array[index] = P_x
            M_array[index] = M_x_sub
            T_array[index] = T_x

        flow_result = FlowResults(x_conv, y_conv, x_div, y_div, P_array,
                                  M_array, T_array, P_e_sup, P_e_sub,
                                  P_e_shock, P_star_shock, m_dot, P_e,
                                  expansion_ratio, m_prop, ISP, thr)

        return flow_result

    def calc_depress(self, UIInputs, FlowResults):
        """
        Calculates flow properties during isothermal or adiabatic
        depressurization. Assumes choked flow and uses derivative eqns. with
        Euler's method.
        """

        # Initial Conditions
        time_step = .00001
        t = 0
        P_0_init = P_0
        T_0_init = T_0
        rho_0_init = PropsSI('D', 'P', P_0_init, 'T', T_0_init, fluid)
        V_init = len_inlet * (np.pi * (r_inlet**2))
        m_init = rho_0_init * V_init
        P_curr = P_0
        T_curr = T_0
        rho_curr = PropsSI('D', 'P', P_curr, 'T', T_curr, fluid)
        m_curr = V_init * rho_curr
        C_d = 1
        
        P_depress_array = []
        m_depress_array = []
        thr_depress_array = []
        temp_depress_array = []

        AS = AbstractState("HEOS", fluid)

        c_0 = np.sqrt(k*R_spec * T_0_init)
        exp = (k + 1) / (2 * k - 2)
        tau = (V_init / (C_d * A_star * c_0))*(((k+1)/2)**exp)
        decay_time = 4.61 * tau         # Time until 1 percent remaining
        if decay_time/1000 > time_step: # Check if timestep too small
            time_step = decay_time/1000

        if depress_type_list.currentIndex() == 0:
            calc_dPdt = lambda P, tau, t: AT.calc_isotherm_dPdt(P, tau, t)
            calc_drhodt = lambda rho, tau, t: AT.calc_isotherm_drhodt(
                rho, tau, t)
            calc_dTdt = lambda T, tau, t: 0
        elif depress_type_list.currentIndex() == 1:
            calc_dPdt = lambda P, tau, t: AT.calc_ada_dPdT(P, tau, t, k)
            calc_drhodt = lambda rho, tau, t: AT.calc_ada_drhodt(
                rho, tau, t, k)
            calc_dTdt = lambda T, tau, t: AT.calc_ada_dTdt(T, tau, t, k)

        # Main depressurization loop
        while m_curr > m_init*0.01 and t < decay_time:
            t += time_step
            dPdt = calc_dPdt(P_0_init, tau, t)
            drhodt = calc_drhodt(rho_0_init, tau, t)
            dTdt = calc_dTdt(T_0_init, tau, t)

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

            P_0 = P_curr
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

    def calc_opt_geom(self):
        """
        Calculates the optimal nozzle geometry using a gradient descent
        algorithm with an ADAM optimizer to match design thrust.
        """

        self.extract_UI_data()
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
