import os
import pandas as pd
from pandas import DataFrame


class DiseasesPrecautions(object):
    def __init__(self):
        curr_dir = os.path.dirname(__file__)
        self.data_path = os.path.join(curr_dir, "data")
        disease_description_path = os.path.join(self.data_path, "symptom_Description.csv")
        disease_precaution_path = os.path.join(self.data_path, "symptom_precaution.csv")
        self.descriptions = self.get_data(pd.read_csv(disease_description_path), "Disease", "Description")
        self.precautions = self.get_data(pd.read_csv(disease_precaution_path), "Disease", [f"Precaution_{i}" for i in
                                                                                           range(1, 5)])

    def get_data(self, data_csv: DataFrame, key_col_name, value_cols_name):
        result = {}

        for index, row in data_csv.iterrows():
            result[row[key_col_name]] = row[value_cols_name]

        return result