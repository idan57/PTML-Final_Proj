from threading import Thread


class TaskResult(object):
    def __init__(self):
        self.verdict = False
        self.result_value = None


class Task(object):
    _thread: Thread
    result: TaskResult

    def __init__(self, name, func, args=None):
        self.name = name
        self._thread = None
        self._func = func
        self._args = args
        self.running = True
        self.result = None

    def _task_execution(self):
        if self._args:
            self.result = self._func(*self._args)
        else:
            self.result = self._func()
        self.running = False

    def start_task(self):
        self._thread = Thread(target=self._task_execution)
        self._thread.start()
