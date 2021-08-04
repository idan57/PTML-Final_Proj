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
        """
        :param import_name: name of the app
        :param args: any arguments for the app
        :param kwargs: any kwargs for the app
        """
        os.chdir(os.path.dirname(__file__))
        super().__init__(import_name, *args, **kwargs)
        self._tasks_container = TasksContainer()
        self.model = None
        self.symptoms = None
        self.diseases_precautions = DiseasesPrecautions()
        self._setup()

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        """
        Start the app
        """
        self.model = DiseasesModel()
        self.model.train()
        self.model.save_model()
        self.symptoms = self.model.symptoms_to_weights.keys()
        self.symptoms = [sym.replace("_", " ") for sym in self.symptoms]
        super().run(host=None, port=None, debug=None, load_dotenv=True, **options)

    def pred_model(self, symptoms: list):
        """
        Make a prediction for given symptoms
        :param symptoms: list of symptoms
        :type symptoms: list
        :return: prediction for a disease
        :rtype: str
        """
        prediction = self.model.predict(symptoms)
        self.logger.info(f"The prediction is: {prediction}")
        res = TaskResult()
        res.result_value = prediction
        res.verdict = True
        return res

    def _setup(self):
        @self.route("/")
        def home():
            """
            :return: Home page
            """
            return flask.render_template("home.html", symptoms=self.symptoms)

        @self.route("/About")
        def about():
            """
            :return: About page
            """
            return flask.render_template("about.html")

        @self.route("/SearchDisease")
        def search():
            """
            :return: Search Disease page
            """
            return flask.render_template("search.html")

        @self.route("/TaskStatus/<task_name>")
        def task_status(task_name):
            """
            :param task_name: task name
            :type task_name: str
            :return: status of the task
            """
            return flask.make_response(str(self._tasks_container.is_task_done(task_name=task_name)))

        @self.route("/TaskResult/<task_name>")
        def task_result(task_name):
            """
            :param task_name: task name
            :type task_name: str
            :return: the task's result page
            """
            sickness = self._tasks_container.get_task_result(task_name)
            description = None
            final_precautions = None
            if sickness.result_value in self.diseases_precautions.descriptions:
                description = self.diseases_precautions.descriptions[sickness.result_value.lower()]
            if sickness.result_value in self.diseases_precautions.precautions:
                precautions = self.diseases_precautions.precautions[sickness.result_value.lower()]
                final_precautions = []
                num_of_precaution = 1
                for key, val in precautions.items():
                    if val:
                        val = str(val)[0].upper() + str(val)[1:]
                        final_precautions.append((f"Precaution {num_of_precaution}", val))
                        num_of_precaution += 1
            return flask.render_template("task_result.html", sickness=sickness, description=description,
                                         precautions=final_precautions)

        @self.route("/SearchDiseaseResult", methods=["GET", "POST"])
        def search_disease_result():
            """
            Submitting data for prediction

            :return: submission page
            """
            if request.form:
                disease_name = request.form["disease_name"]
                dis_lower = disease_name.lower()
                if dis_lower in self.diseases_precautions.descriptions:
                    description = self.diseases_precautions.descriptions[dis_lower]
                    precautions = None
                    if dis_lower in self.diseases_precautions.precautions:
                        precautions = self.diseases_precautions.precautions[dis_lower]
                        prs_only = []
                        for prec in precautions:
                            prs_only.append(prec)
                        precautions = ", ".join(prs_only)
                    return flask.render_template("search_result.html",
                                                 description=description,
                                                 disease_name=disease_name,
                                                 precautions=precautions,
                                                 invalid_disease=None)
                else:
                    return flask.render_template("search_result.html", invalid_disease=disease_name)
            return flask.render_template("submit_results.html")

        @self.route("/SubmitMedicalResults", methods=["GET", "POST"])
        def submit_data():
            """
            Submitting data for prediction

            :return: submission page
            """
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
