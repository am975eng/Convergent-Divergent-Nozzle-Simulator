import unittest
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import dataclass
from models.thruster import ThrusterModel


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
    depress_type: int


logging.disable(logging.CRITICAL)


class TestCDNozzle(unittest.TestCase):
    def setUp(self):
        self.model = ThrusterModel()
        fluid = "Air"
        R_spec = 287
        k = 1.4
        T_0 = 300
        P_0 = 1e5
        rho_0 = P_0 / (R_spec * T_0)
        P_amb = 0
        T_star = T_0 * (2 / (k + 1)) ** (k / (k - 1))
        P_star = P_0 * (2 / (k + 1)) ** (k / (k - 1))
        M_star = 1
        r_throat = 0.5
        r_inlet = 0.6
        r_outlet = 0.75
        converg_angle = 30
        diverg_angle = 30
        len_inlet = 1
        M_exit_moc = 1.5
        thr_design = 100
        noz_type = "Conical"
        depress_type = "Adiabatic"

        self.UI_input = UIInputs(
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
        )

    def test_thermo_vacuum(self):
        # Vacuum back pressure test
        self.UI_input.P_amb = 0
        self.UI_input, flow_result = self.model.calc_thermo(self.UI_input)

        # Check if exit pressure match external isentropic calculator
        self.assertAlmostEqual(
            flow_result.P_e_sup,
            self.UI_input.P_0 * 0.0765,
            delta=self.UI_input.P_0 * 0.0765 / 100,
        )
        self.assertAlmostEqual(
            flow_result.P_e_sub,
            self.UI_input.P_0 * 0.95,
            delta=self.UI_input.P_0 * 0.95 / 100,
        )

        self.assertGreater(flow_result.P_e, self.UI_input.P_amb)
        # A shock at throat should yield higher pressure than exit shock
        self.assertGreater(flow_result.P_star_shock, flow_result.P_e_shock)

    def test_thermo_shock(self):
        # Check if normal shock can be handled
        self.UI_input, flow_result = self.model.calc_thermo(self.UI_input)
        self.UI_input.P_amb = (
            flow_result.P_star_shock + flow_result.P_e_shock
        ) / 2
        self.UI_input, flow_result = self.model.calc_thermo(self.UI_input)

        # Check if exit pressure is close to ambient pressure
        self.assertAlmostEqual(
            flow_result.P_e, self.UI_input.P_amb, delta=flow_result.P_e / 100
        )

    def test_opt_geom_min(self):
        self.UI_input.r_throat = 1e-5
        self.UI_input.r_inlet = 1e-4
        self.UI_input.r_outlet = 1e-4

        self.UI_input, flow_result = self.model.calc_thermo(self.UI_input)
        self.assertIsNotNone(flow_result)
        self.UI_input.thr_design = 100
        for opt_output in self.model.calc_opt_geom(
            self.UI_input, flow_result, 1000
        ):
            self.UI_input, flow_result, itx = opt_output
            print(itx)
            print(flow_result.thr)
        self.assertAlmostEqual(
            flow_result.thr,
            self.UI_input.thr_design,
            delta=self.UI_input.thr_design / 100,
        )

    # def test_opt_geom_max(self):
    #     self.UI_input.r_throat = 1e5
    #     self.UI_input.r_inlet = 1e6
    #     self.UI_input.r_outlet = 1e6

    #     self.UI_input, flow_result = self.model.calc_thermo(self.UI_input)
    #     self.assertIsNotNone(flow_result)
    #     try:
    #         while True:
    #             opt_output = next(self.model.calc_opt_geom(self.UI_input, flow_result))
    #     except StopIteration:
    #         self.UI_input, flow_result = opt_output
    #     self.assertAlmostEqual(flow_result.thr, self.UI_input.thr_design, delta=self.UI_input.thr_design / 100)


if __name__ == "__main__":
    unittest.main()
