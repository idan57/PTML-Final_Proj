import logging
from threading import Thread


class TaskResult(object):
    """
    A class for a task's result
    """
    def __init__(self):
        self.verdict = False
        self.result_value = None


class Task(object):
    """
    A class to represent a task
    """
    _thread: Thread
    result: TaskResult

    def __init__(self, name, func, args=None):
        """
        :param name: name of the task
        :param func: the method for executing the class
        :param args: any arguments needed to start the task
        """
        self.name = name
        self._thread = None
        self._func = func
        self._args = args
        self.running = True
        self.result = None

    def _task_execution(self):
        logging.info(f"{self.name} is now running")
        if self._args:
            self.result = self._func(*self._args)
        else:
            self.result = self._func()
        self.running = False
        logging.info(f"{self.name} is now done!")

    def start_task(self):
        """
        Start executing the task
        """
        logging.info(f"Starting task: {self.name}")
        self._thread = Thread(target=self._task_execution)
        self._thread.start()
        logging.info(f"Started task: {self.name}")
