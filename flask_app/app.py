import time

import flask
from flask import Flask, request
import os

from data_containers.task import Task, TaskResult
from data_containers.tasks_container import TasksContainer
from model.diseases_model import DiseasesModel
from model.diseases_precautions import DiseasesPrecautions


def run_something():
    time.sleep(10)
    res = TaskResult()
    res.result_value = "You are all good!"
    res.verdict = False
    return res


class MainApp(Flask):
    _tasks_container: TasksContainer
    model: DiseasesModel

    def __init__(self, import_name="MainApp", *args, **kwargs):
        os.chdir(os.path.dirname(__file__))
        super().__init__(import_name, *args, **kwargs)
        self._tasks_container = TasksContainer()
        self.model = None
        self.symptoms = None
        self.diseases_precautions = DiseasesPrecautions()
        self._setup()

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        self.model = DiseasesModel(app=self)
        self.model.train()
        self.model.save_model()
        self.symptoms = self.model.symptoms_to_weights.keys()
        self.symptoms = [sym.replace("_", " ") for sym in self.symptoms]
        super().run(host=None, port=None, debug=None, load_dotenv=True, **options)

    def pred_model(self, symptoms):
        prediction = self.model.predict(symptoms)
        self.logger.info(f"The prediction is: {prediction}")
        res = TaskResult()
        res.result_value = prediction
        res.verdict = True
        return res

    def _setup(self):
        @self.route("/")
        def home():
            return flask.render_template("home.html", symptoms=self.symptoms)

        @self.route("/About")
        def about():
            return flask.render_template("about.html")

        @self.route("/TaskStatus/<task_name>")
        def task_status(task_name):
            return flask.make_response(str(self._tasks_container.is_task_done(task_name=task_name)))

        @self.route("/TaskResult/<task_name>")
        def task_result(task_name):
            sickness = self._tasks_container.get_task_result(task_name)
            description = None
            final_precautions = None
            if sickness.result_value in self.diseases_precautions.descriptions:
                description = self.diseases_precautions.descriptions[sickness.result_value]
            if sickness.result_value in self.diseases_precautions.precautions:
                precautions = self.diseases_precautions.precautions[sickness.result_value]
                final_precautions = []
                num_of_precaution = 1
                for key, val in precautions.items():
                    if val:
                        val = str(val)[0].upper() + str(val)[1:]
                        final_precautions.append((f"Precaution {num_of_precaution}", val))
                        num_of_precaution += 1
            return flask.render_template("task_result.html", sickness=sickness, description=description,
                                         precautions=final_precautions)

        @self.route("/SubmitMedicalResults", methods=["GET", "POST"])
        def submit_data():
            if request.form:
                symptoms = request.form["symptoms"].replace("\r", "").split("\n")
                form = dict(request.form)
                form["symptoms"] = ", ".join(symptoms)
                symptoms = [symptom.replace(" ", "_") for symptom in symptoms]
                task = Task(name=request.form["name"], func=self.pred_model, args=(symptoms, ))
                self._tasks_container.add_task(task=task)
                return flask.render_template("submit_results.html",
                                             form_items=form.items(),
                                             task_name=task.name)
            return flask.render_template("submit_results.html")
