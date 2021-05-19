from typing import Dict

from data_containers.task import Task


class TaskNotFoundException(Exception):
    pass


class TasksContainer(object):
    _container: Dict[str, Task]

    def __init__(self):
        self._container = {}

    def add_task(self, task: Task):
        task.start_task()
        self._container[task.name] = task

    def is_task_done(self, task_name):
        if task_name in self._container:
            return self._container[task_name].running
        raise TaskNotFoundException(f"Could not find task: {task_name}")
