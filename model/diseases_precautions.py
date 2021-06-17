import os
import pandas as pd
from pandas import DataFrame


class DiseasesPrecautions(object):
    """
    Class to contain all disease precautions and descriptions
    """

    def __init__(self):
        curr_dir = os.path.dirname(__file__)
        self.data_path = os.path.join(curr_dir, "data")
        disease_description_path = os.path.join(self.data_path, "symptom_Description.csv")
        disease_precaution_path = os.path.join(self.data_path, "symptom_precaution.csv")
        self.descriptions = self._get_data(pd.read_csv(disease_description_path).fillna(0), "Disease", "Description")
        self.precautions = self._get_data(pd.read_csv(disease_precaution_path).fillna(0), "Disease",
                                          [f"Precaution_{i}" for i in range(1, 5)])

    @classmethod
    def _get_data(cls, data_csv: DataFrame, key_col_name, value_cols_name):
        result = {}

        for index, row in data_csv.iterrows():
            result[row[key_col_name]] = row[value_cols_name]

        return result
