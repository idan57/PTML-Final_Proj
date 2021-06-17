from typing import Dict

from data_containers.task import Task


class TaskNotFoundException(Exception):
    """
    An exception class for none existing tasks in the tasks container
    """
    pass


class TasksContainer(object):
    """
    A class that contains all tasks
    """
    _container: Dict[str, Task]

    def __init__(self):
        self._container = {}

    def add_task(self, task: Task):
        """
        Add new task to container
        :param task: task to add
        """
        task.start_task()
        self._container[task.name] = task

    def get_task_result(self, task_name):
        """
        Return a task's result

        :param task_name: task name
        :return: the task's result
        """
        if task_name in self._container:
            return self._container[task_name].result
        raise TaskNotFoundException(f"Could not find task: {task_name}")

    def is_task_done(self, task_name):
        """
        Return if the task is done
        :param task_name: task's name
        :return: true if task is done
        """
        if task_name in self._container:
            return self._container[task_name].running
        raise TaskNotFoundException(f"Could not find task: {task_name}")
