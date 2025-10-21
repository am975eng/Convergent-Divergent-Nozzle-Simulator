import inspect

from PyQt6.QtCore import QRunnable, QObject, pyqtSignal


class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    progress = pyqtSignal(object)


class Worker(QRunnable):
    """Task to be executed in a separate thread."""
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        # If the function is a generator we need to emit progress signals
        if inspect.isgeneratorfunction(self.fn):
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
