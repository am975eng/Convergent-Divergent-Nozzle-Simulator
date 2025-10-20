from PyQt6.QtCore import QRunnable, QObject, pyqtSignal, QThread


class WorkerSignals(QObject):
    finished = pyqtSignal(object)
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
        if 'progress_callback' in self.fn.__code__.co_varnames:
            gen = self.fn(*self.args, **self.kwargs)
            while True:
                try:
                    partial_result = next(gen)
                    self.signals.progress.emit(partial_result)
                except StopIteration as e:
                    final_result = e.value
                    self.signals.finished.emit(final_result)
                    break
        else:
            try:
                result = self.fn(*self.args, **self.kwargs)
            except Exception as e:
                self.signals.error.emit(str(e))
            else:
                self.signals.result.emit(result)
