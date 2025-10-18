from PyQt6.QtCore import QTimer, QThreadPool
from workers import Worker

class ThrusterController:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.threadpool = QThreadPool()


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

        self.view.optimize_button.clicked.connect(self.optimize_geom)

        self.view.depress_button.clicked.connect(self.calc_depress)

        self.update_result()


    def schedule_update(self):
        # Start the debounce timer to prevent rapid updates
        self.debounce.start()

    def update_result(self):
        UI_input = self.view.extract_UI_data()
        worker = Worker(self.model.calc_thermo, UI_input)
        
        worker.signals.result.connect(self.on_results_ready)
        worker.signals.finished.connect(self.on_finished)
        worker.signals.error.connect(self.on_error)
        self.threadpool.start(worker)
        self.view.show_busy(True)

    def on_finished(self):
        print("Finished")

    def on_error(self, msg):
        self.view.show_error(msg)
        self.view.show_busy(False)
    
    def on_results_ready(self, UI_input, flow_result):
        self.view.plot_data(UI_input, flow_result)
        self.view.show_busy(False)

    def optimize_geom(self):
        UI_input = self.view.extract_UI_data()
        thermo_worker = Worker(self.model.calc_thermo, UI_input)
        thermo_worker.signals.result.connect(self.on_thermo_ready)
        thermo_worker.signals.finished.connect(lambda: print("Thermo done"))
        self.threadpool.start(thermo_worker)

    def on_thermo_ready(self, UI_input, flow_result):
        opt_worker = Worker(self.model.calc_opt_geom, UI_input, flow_result, 1000, 1E-3)
        opt_worker.signals.result.connect(self.on_opt_done)
        opt_worker.signals.finished.connect(lambda: print("Optimization finished"))
        self.threadpool.start(opt_worker)

    def on_opt_done(self):
        pass

    def calc_depress(self):
        pass
