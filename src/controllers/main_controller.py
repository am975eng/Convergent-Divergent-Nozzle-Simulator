
from views.main_view import MainWindow
from models.thruster import ThrusterModel
from PyQt6.QtCore import Qt, QTimer

class ThrusterController:
    def __init__(self, view, model):
        self.view = view
        self.model = model

        self.view.optimize_button.clicked.connect(self.calc_opt_geom)

        self.debounce = QTimer(singleShot=True, interval=400)
        self.debounce.timeout.connect(self.update_result)

        self.view.prop_list.currentTextChanged.connect(self.schedule_update)
        self.view.noz_type_list.currentTextChanged.connect(self.schedule_update)
        self.view.P_chamber_val.textChanged.connect(self.schedule_update)
        self.view.T_chamber_val.textChanged.connect(self.schedule_update)
        self.view.converg_ang_val.textChanged.connect(self.schedule_update)
        self.view.diverg_angle_val.textChanged.connect(self.schedule_update)
        self.view.length_inlet_val.textChanged.connect(self.schedule_update)
        self.view.radius_inlet_val.textChanged.connect(self.schedule_update)
        self.view.radius_throat_val.textChanged.connect(self.schedule_update)
        self.view.radius_exit_val.textChanged.connect(self.schedule_update)
        self.view.M_exit_val.textChanged.connect(self.schedule_update)
        self.view.P_amb_val.textChanged.connect(self.schedule_update)
        self.view.depress_button.clicked.connect(self.calc_depress)

        self.update_result()


    def schedule_update(self):
        # Start the debounce timer to prevent rapid updates
        self.debounce.start()

    def update_result(self):
        UI_input = self.view.extract_UI_data()
        print(UI_input)
        flow_result = self.model.calc_thermo(UI_input)

    def calc_opt_geom(self):
        pass

    def calc_depress(self):
        pass
