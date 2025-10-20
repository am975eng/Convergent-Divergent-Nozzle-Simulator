from PyQt6.QtCore import QRunnable, QObject, pyqtSignal, QThread

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)   # will emit UI_input and FlowResults
    progress = pyqtSignal(object)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            if 'progress_callback' in self.fn.__code__.co_varnames:
                for partial_result in self.fn(*self.args, **self.kwargs):
                    self.signals.progress.emit(partial_result)
                self.signals.finished.emit(partial_result)
               #result = self.fn(*self.args, progress_callback=self.signals.progress.emit, **self.kwargs)
            else:
               result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit(str(e))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
