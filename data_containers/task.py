from threading import Thread


class Task(object):
    _thread: Thread

    def __init__(self, name, func, args):
        self.name = name
        self._thread = None
        self._func = func
        self._args = args
        self.running = True

    def _task_execution(self):
        self._func(*self._args)
        self.running = False

    def start_task(self):
        self._thread = Thread(target=self._task_execution)
        self._thread.start()
