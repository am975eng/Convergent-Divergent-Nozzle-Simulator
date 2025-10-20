import sys
from views.main_view import MainWindow
from models.thruster import ThrusterModel
from controllers.main_controller import ThrusterController
from PyQt6.QtWidgets import QApplication


app = QApplication(sys.argv)
view = MainWindow()
model = ThrusterModel()
controller = ThrusterController(view, model)
controller.view.show()
sys.exit(app.exec())