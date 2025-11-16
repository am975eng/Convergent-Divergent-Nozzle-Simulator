from PyQt6.QtCore import QTimer, QThreadPool, QObject

from controllers.workers import Worker


class ThrusterController(QObject):
    """Controller used to handle user input and signal the model and view to
    update accordingly.

    Creates worker threads to run complex calculations in the background while
    the UI remains responsive. Contains methods for triggering model
    calculations based on user action.
    """

    def __init__(self, view, model):
        super().__init__()
        self.view = view
        self.model = model
        self.threadpool = QThreadPool()

        self.debounce = QTimer(singleShot=True, interval=500)
        self.debounce.timeout.connect(self.set_calc_needed)

        self.view.prop_list.currentTextChanged.connect(self.schedule_update)
        self.view.noz_type_list.currentTextChanged.connect(self.schedule_update)
        self.view.P_chamber_val.textChanged.connect(self.schedule_update)
        self.view.T_chamber_val.textChanged.connect(self.schedule_update)
        self.view.converg_ang_val.textChanged.connect(self.schedule_update)
        self.view.diverg_angle_val.textChanged.connect(self.schedule_update)
        self.view.length_inlet_val.textChanged.connect(self.schedule_update)
        self.view.radius_inlet_val.textChanged.connect(self.schedule_update)
        self.view.radius_throat_val.textChanged.connect(self.schedule_update)
        self.view.radius_outlet_val.textChanged.connect(self.schedule_update)
        self.view.M_exit_moc_val.textChanged.connect(self.schedule_update)
        self.view.P_amb_val.textChanged.connect(self.schedule_update)

        self.view.optimize_button.clicked.connect(self.optimize_geom)

        self.view.depress_button.clicked.connect(self.run_depress)

        self.view.monte_carlo_button.clicked.connect(self.run_monte_carlo)

        self.view.calc_button.clicked.connect(self.update_result)

        self.view.update_UI_nozzle()

        self.update_result()

    def schedule_update(self):
        # Start the debounce timer to prevent rapid updates
        self.debounce.start()

    def set_calc_needed(self):
        self.view.update_UI_nozzle()
        self.view.calc_button.setStyleSheet("background-color: red;")

    def update_result(self):
        self.view.calc_button.setStyleSheet("background-color: blue;")
        UI_input = self.view.extract_UI_data()
        worker = Worker(self.model.calc_thermo, UI_input)
        worker.signals.result.connect(self.on_results_ready)
        worker.signals.error.connect(self.on_error)
        self.threadpool.start(worker)

    def on_error(self, msg):
        print(msg)

    def on_results_ready(self, result):
        UI_input, flow_result = result
        self.view.plot_flow_data(UI_input, flow_result)
        self.view.calc_button.setStyleSheet("background-color: green;")

    def optimize_geom(self):
        self.view.set_busy_state("optimize")
        UI_input = self.view.extract_UI_data()
        UI_input, flow_result = self.model.calc_thermo(UI_input)
        opt_worker = Worker(
            self.model.calc_opt_geom, UI_input, flow_result, 1000
        )
        opt_worker.signals.progress.connect(self.on_optimize_progress)
        opt_worker.signals.finished.connect(self.on_optimize_finished)
        self.threadpool.start(opt_worker)

    def on_optimize_progress(self, opt_data_update):
        UI_input, flow_result, bar_value = opt_data_update
        self.view.plot_flow_data(UI_input, flow_result, bar_value)

    def on_optimize_finished(self, opt_result):
        UI_input, flow_result = opt_result
        self.view.plot_flow_data(UI_input, flow_result)
        self.view.set_busy_state("finished")

    def run_depress(self):
        self.view.set_busy_state("depress")
        UI_input = self.view.extract_UI_data()
        result = self.model.calc_thermo(UI_input)
        UI_input, flow_result = result
        depress_worker = Worker(self.model.calc_depress, UI_input, flow_result)
        depress_worker.signals.progress.connect(self.on_depress_progress)
        depress_worker.signals.finished.connect(self.on_depress_finished)
        self.threadpool.start(depress_worker)

    def on_depress_progress(self, depress_data_update):
        self.view.plot_depress_update(depress_data_update)

    def on_depress_finished(self, depress_result):
        self.view.plot_depress_final(depress_result)
        self.view.set_busy_state("finished")

    def run_monte_carlo(self):
        self.view.set_busy_state("monte_carlo")
        UI_input = self.view.extract_UI_data()
        result = self.model.calc_thermo(UI_input)
        UI_input, flow_result = result
        mc_worker = Worker(self.model.calc_monte_carlo, UI_input, flow_result)
        mc_worker.signals.result.connect(self.on_mc_finished)
        self.threadpool.start(mc_worker)

    def on_mc_finished(self, outputs):
        print("Finished")
        print(outputs)
        self.view.set_busy_state("finished")