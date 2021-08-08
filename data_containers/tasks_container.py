import logging
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
        logging.info(f"Adding the task '{task.name}' to the Tasks Container")
        task.start_task()
        self._container[task.name] = task

    def get_task_result(self, task_name):
        """
        Return a task's result

        :param task_name: task name
        :return: the task's result
        """
        logging.info(f"Getting task: {task_name}")
        if task_name in self._container:
            logging.info("Success!")
            return self._container[task_name].result
        logging.error(f"Could not find task: {task_name}")
        raise TaskNotFoundException(f"Could not find task: {task_name}")

    def is_task_done(self, task_name):
        """
        Return if the task is done
        :param task_name: task's name
        :return: true if task is done
        """
        logging.info(f"Checking if '{task_name}' is done")
        if task_name in self._container:
            res = self._container[task_name].running
            logging.info(f"Result: {res}")
            return res
        logging.info(f"Could not find task: {task_name}")
        raise TaskNotFoundException(f"Could not find task: {task_name}")
