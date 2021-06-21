import json
import os
import pickle
from collections import Counter
from multiprocessing import Pool
import multiprocessing
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

tune_options = {
    "DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": [None, 10, 20, 30]
    },
    "RandomForestClassifier": {
        "n_estimators": [100, 500, 1000],
        "criterion": ["gini", "entropy"],
        "warm_start": [True, False]
    },
    "SVC": {
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
        "C": [0.1, 1, 10]
    },
    "KNN": {
        "n_neighbors": [5, 7, 11],
        "weights": ["uniform", "distance"],
    },
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "solver": ["newton-cg", "lbfgs", "saga"],
        "warm_start": [True, False],
        "max_iter": [100, 120, 150]
    },
    "AdaBoostClassifier": {
        "learning_rate": [0.01, 0.9, 1],
        "n_estimators": [50, 100]
    }

}


class PickledModelDoesNotExistException(Exception):
    """
    Exception class that will be thrown in case we don't find some pickled file for a model
    """
    pass


class DiseasesModel(object):
    """
    A model for training and fitting the diseases data
    """

    def __init__(self):
        """
        :param app: the flask app for logging purposes
        :type app: Flask
        """
        curr_dir = os.path.dirname(__file__)
        self.data_path = os.path.join(curr_dir, "data")
        self.models_path = os.path.join(curr_dir, "models")
        dataset_csv = os.path.join(self.data_path, "dataset.csv")
        symptom_severity_csv = os.path.join(self.data_path, "Symptom-severity.csv")
        self.diseases_data = pd.read_csv(dataset_csv).fillna(0)
        self.symptom_severity_data = pd.read_csv(symptom_severity_csv).fillna(0)
        self.symptoms_to_weights = {}
        self.symptoms_found_in_cols = {}
        self.num_of_symptoms = 0
        self.diseases = []
        self.num_of_diseases = 0
        self.max_weight = 0
        self.zero_mappings = {}
        self._tuned = {}

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # Models paths
        self.decision_tree_path = os.path.join(self.models_path, "DecisionTreeClassifier")
        self.bagging_path = os.path.join(self.models_path, "BaggingClassifier")
        self.random_forest_path = os.path.join(self.models_path, "RandomForestClassifier")
        self.svc_path = os.path.join(self.models_path, "SVC")
        self.knn_path = os.path.join(self.models_path, "KNN")
        self.lr_path = os.path.join(self.models_path, "LogisticRegression")
        self.ada_path = os.path.join(self.models_path, "AdaBoostClassifier")

        self.decision_tree_model = None
        self.bagging_model = None
        self.random_forest_model = None
        self.svc_model = None
        self.knn_model = None
        self.logistic_regression_model = None
        self.ada_model = None

        self._setup()

    def train(self):
        """
        Training our model
        """
        try:
            self._load_models_from_files()
            classifiers = [self.svc_model, self.decision_tree_model, self.bagging_model, self.random_forest_model,
                           self.ada_model, self.logistic_regression_model, self.knn_model]
            for clasiffier in classifiers:
                self.log_train_res(clasiffier)
        except PickledModelDoesNotExistException:
            print("No models were trained in the past, starting training...")
            self._tune()
            self._train()

    def _tune_one(self, classifier, classifier_name):
        print(f"---------- Started tuning with {classifier_name} ----------")
        grid_search_cv = RandomizedSearchCV(classifier,
                                            tune_options[classifier_name],
                                            cv=5,
                                            scoring="accuracy",
                                            verbose=2,
                                            n_jobs=5)
        grid_search_cv.fit(self.x_train, self.y_train)
        print(f"---------- Ended tuning with {classifier_name} ----------")
        return grid_search_cv.best_params_, classifier_name

    def _tune(self):
        tuned_data_json = os.path.join(os.path.dirname(__file__), "tuned_data.json")
        if os.path.isfile(tuned_data_json):
            with open(tuned_data_json, "r") as f:
                self._tuned = json.load(f)
        else:
            procs = []
            classifiers = {
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "SVC": SVC(),
                "KNN": KNeighborsClassifier(),
                "LogisticRegression": LogisticRegression(multi_class="multinomial"),
                "AdaBoostClassifier": AdaBoostClassifier(),
            }
            pool = Pool(processes=3)
            for classifier_name, classifier in classifiers.items():
                procs.append((classifier_name, pool.apply_async(self._tune_one, (classifier, classifier_name))))

            while True:
                i = 0
                num_procs = len(procs)
                while i < num_procs:
                    try:
                        best_res, classifier_name = procs[i][1].get(1)
                        num_procs -= 1
                        del procs[i]
                        self._tuned[classifier_name] = best_res
                    except multiprocessing.TimeoutError:
                        print(f"Still waiting... remaining {num_procs} processes.\nCurrently checked: {procs[i][0]}")
                    i += 1

                if num_procs == 0:
                    break
            with open(tuned_data_json, "w") as f:
                json.dump(self._tuned, f, sort_keys=True, indent=4)

    def predict(self, symptoms: list) -> str:
        """
        Predicting the disease by given symptoms

        :param symptoms: symptoms for a disease
        :type symptoms: list
        :return: prediction for the disease
        :rtype: str
        """
        symptoms = set(symptoms)
        symptoms_variations = self._get_symptoms_variations(symptoms)

        values = self.symptoms_to_vals(symptoms=symptoms_variations)
        models = [self.decision_tree_model, self.bagging_model, self.random_forest_model, self.svc_model,
                  self.ada_model, self.knn_model, self.logistic_regression_model]
        predictions = []

        for model in models:
            prediction = model.predict(values)[0]
            predictions.append(prediction)

        counted_predictions = Counter(predictions)
        top_prediction = max(counted_predictions, key=counted_predictions.get)
        return top_prediction

    def _get_symptoms_variations(self, symptoms):
        combinations = [0 for _ in range(17)]
        for symptom in symptoms:
            options = self.symptoms_found_in_cols[symptom]
            for op in options:
                op_i = int(op.split("_")[1]) - 1
                if combinations[op_i] == 0 or symptom not in combinations:
                    combinations[op_i] = symptom
        return combinations

    def save_model(self):
        """
        Pickles all model into files
        """
        self._pickle(self.decision_tree_path, self.decision_tree_model)
        self._pickle(self.random_forest_path, self.random_forest_model)
        self._pickle(self.bagging_path, self.bagging_model)
        self._pickle(self.svc_path, self.svc_model)
        self._pickle(self.knn_path, self.knn_model)
        self._pickle(self.lr_path, self.logistic_regression_model)
        self._pickle(self.ada_path, self.ada_model)

    def _load_models_from_files(self):
        paths = [self.decision_tree_path, self.bagging_path, self.random_forest_path, self.svc_path,
                 self.knn_path, self.ada_path, self.lr_path]

        for path in paths:
            if not os.path.isfile(path):
                raise PickledModelDoesNotExistException(f"The following pickled file model: {path}, doesn't exist")

        self.decision_tree_model = DiseasesModel._get_unpickled(self.decision_tree_path)
        self.bagging_model = DiseasesModel._get_unpickled(self.bagging_path)
        self.random_forest_model = DiseasesModel._get_unpickled(self.random_forest_path)
        self.svc_model = DiseasesModel._get_unpickled(self.svc_path)
        self.knn_model = DiseasesModel._get_unpickled(self.knn_path)
        self.logistic_regression_model = DiseasesModel._get_unpickled(self.lr_path)
        self.ada_model = DiseasesModel._get_unpickled(self.ada_path)

    @staticmethod
    def _get_unpickled(path: str):
        with open(path, "rb") as pickled_file:
            res = pickle.load(pickled_file)
        return res

    def _pickle(self, path: str, model):
        if model:
            with open(path, "wb") as pickled_file:
                pickle.dump(model, pickled_file)
        else:
            print(f"Requested pickling to {path} was canceled because the model is None")

    def _train(self):
        self.decision_tree_model = DecisionTreeClassifier(**self._tuned["DecisionTreeClassifier"])
        self.bagging_model = BaggingClassifier(n_jobs=3)
        self.random_forest_model = RandomForestClassifier(**self._tuned["RandomForestClassifier"])
        self.svc_model = SVC(**self._tuned["SVC"])
        self.knn_model = KNeighborsClassifier(**self._tuned["KNN"])
        self.logistic_regression_model = LogisticRegression(multi_class='multinomial',
                                                            **self._tuned["LogisticRegression"])
        self.ada_model = AdaBoostClassifier(**self._tuned["AdaBoostClassifier"])

        self.decision_tree_model.fit(self.x_train, self.y_train)
        self.log_train_res(self.decision_tree_model)

        self.bagging_model.fit(self.x_train, self.y_train)
        self.log_train_res(self.bagging_model)

        self.random_forest_model.fit(self.x_train, self.y_train)
        self.log_train_res(self.random_forest_model)

        self.svc_model.fit(self.x_train, self.y_train)
        self.log_train_res(self.svc_model)

        self.knn_model.fit(self.x_train, self.y_train)
        self.log_train_res(self.knn_model)

        self.logistic_regression_model.fit(self.x_train, self.y_train)
        self.log_train_res(self.logistic_regression_model)

        self.ada_model.fit(self.x_train, self.y_train)
        self.log_train_res(self.ada_model)

    def log_train_res(self, model):
        from sklearn.metrics import plot_confusion_matrix
        pred_train = model.predict(self.x_train)
        pred_test = model.predict(self.x_test)

        print(f"Train accuracy for {model.__class__.__name__}: {accuracy_score(self.y_train, pred_train)}")
        print(f"Test accuracy for {model.__class__.__name__}: {accuracy_score(self.y_test, pred_test)}")

        print(f"Test Confusion Matrix for {model.__class__.__name__}:\n{confusion_matrix(self.y_test, pred_test)}")
        fig, ax = plt.subplots(figsize=(30, 30))
        display_labels = [f"{i}" for i in range(len(self.diseases))]
        disp = plot_confusion_matrix(model, self.x_test, self.y_test, display_labels=display_labels, ax=ax,
                                     cmap=plt.cm.Blues)
        disp.ax_.set_title(f"Confusion Matrix - {model.__class__.__name__}")
        fig.savefig(model.__class__.__name__)

    def symptoms_to_vals(self, symptoms: List):
        """
        Fitting the symptoms into numbered values

        :param symptoms: symptoms to encode
        :type symptoms: list
        :return: encoded symptoms
        :rtype: np.ndarray
        """
        print(f"Received symptoms: {symptoms}")
        for i in range(len(symptoms)):
            if symptoms[i]:
                symptoms[i] = self.symptoms_to_weights[symptoms[i]][f"Symptom_{i + 1}"]

        return np.asarray(symptoms).reshape(1, -1)

    def _setup(self):
        self._init_symptoms()
        self._init_diseases()
        self._fit_data()
        self._prepare_for_train()

    def _init_symptoms(self):
        for index, row in self.symptom_severity_data.iterrows():
            symptom = row["Symptom"].strip().replace(" ", "")
            weight = row["weight"]
            self.symptoms_found_in_cols[symptom] = set({})
            self.symptoms_to_weights[symptom] = {f"Symptom_{i}": weight for i in range(1, 18)}
            if weight > self.max_weight:
                self.max_weight = weight

        self.num_of_symptoms = len(self.symptoms_to_weights)

        print(f"Symptoms: {self.symptoms_to_weights}")
        print(f"Number of Symptoms: {self.num_of_symptoms}")
        print(f"Max Weight: {self.max_weight}")

    def _init_diseases(self):
        diseases = set()
        for disease in self.diseases_data["Disease"]:
            diseases.add(disease)

        self.diseases = list(diseases)
        self.num_of_diseases = len(self.diseases)

        print(f"Diseases: {self.diseases}")
        print(f"Number of Diseases: {self.num_of_diseases}")

    def _fit_data(self):
        encoder = LabelEncoder()
        col_mappings = {}

        for col in self.diseases_data.columns:
            if col != "Disease":
                row_weight_mapping = []
                for sym in self.diseases_data[col]:
                    # We passed all symptoms
                    if sym == 0:
                        row_weight_mapping += [[0, 0]]
                        continue

                    stripped_sym = sym.strip()
                    if stripped_sym in self.symptoms_to_weights:
                        row_weight_mapping += [[sym, self.symptoms_to_weights[sym.strip()][col]]]
                    else:
                        row_weight_mapping += [[sym, 1]]

                col_mappings[col] = row_weight_mapping
                dis = self.diseases_data[col].astype(str)
                self.diseases_data[col] = encoder.fit_transform(dis)

        for index, row in self.diseases_data.iterrows():
            for col, mapping in col_mappings.items():
                val = row[col]
                if mapping[index][1] == 0:
                    val = 0
                if mapping[index][0]:
                    symptom = mapping[index][0].strip().replace(" ", "")
                    if symptom not in self.symptoms_found_in_cols:
                        self.symptoms_found_in_cols[symptom] = set({})

                    if symptom not in self.symptoms_to_weights:
                        self.symptoms_to_weights[symptom] = {}

                    self.symptoms_found_in_cols[symptom].add(col)
                    self.symptoms_to_weights[symptom][col] = int(val)
                self.diseases_data.at[index, col] = val

        self.num_of_symptoms = len(self.symptoms_to_weights)
        print(f"Symptoms: {self.symptoms_to_weights}")
        print(f"Number of Symptoms: {self.num_of_symptoms}")
        print("Finished fitting data...")

    def _prepare_for_train(self):
        x = self.diseases_data[[f"Symptom_{i}" for i in range(1, 18)]]
        y = self.diseases_data["Disease"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=999)
