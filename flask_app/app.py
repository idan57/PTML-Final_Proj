import time

import flask
from flask import Flask, request
import os

from data_containers.task import Task, TaskResult
from data_containers.tasks_container import TasksContainer


def run_something():
    time.sleep(10)
    res = TaskResult()
    res.result_value = "You are all good!"
    res.verdict = False
    return res


class MainApp(Flask):
    _tasks_container: TasksContainer

    def __init__(self, import_name="MainApp", *args, **kwargs):
        os.chdir(os.path.dirname(__file__))
        super().__init__(import_name, *args, **kwargs)
        self._tasks_container = TasksContainer()
        self._setup()

    def _setup(self):
        @self.route("/")
        def home():
            return flask.render_template("home.html")

        @self.route("/About")
        def about():
            return flask.render_template("about.html")

        @self.route("/TaskStatus/<task_name>")
        def task_status(task_name):
            return flask.make_response(str(self._tasks_container.is_task_done(task_name=task_name)))

        @self.route("/TaskResult/<task_name>")
        def task_result(task_name):
            return flask.render_template("task_result.html", result=self._tasks_container.get_task_result(task_name))

        @self.route("/SubmitMedicalResults", methods=["GET", "POST"])
        def submit_data():
            if request.form:
                task = Task(name=request.form["Name"], func=run_something)
                self._tasks_container.add_task(task=task)
                return flask.render_template("submit_results.html",
                                             form_items=request.form.items(),
                                             task_name=task.name)
            return flask.render_template("submit_results.html")
