import os
import pickle

import numpy as np
import pandas as pd
from pygam import LinearGAM

import mlflow

class ForecastingModel(mlflow.pyfunc.PythonModel):
    """
    ForecastingModel class here is to fit and predict individual forecasting model
    """

    def init(self):
        super().__init__()

    def fit(self, data_input):
        """
        Store Date Sales StateHoliday SchoolHoliday - input columns by design choice
        """
        # here we are accepting the fact that our input might be a numpy array
        # keep your input ordered, Pandas DF are not like Spark and they dont work on a schema but on numbering
        model_input = pd.DataFrame(
            data_input,
            columns=["Store", "Horizon", "Date", "Sales", "StateHoliday", "SchoolHoliday"],
        )
        # becasue it's forecasting I sort per date
        model_input.sort_values("Date", inplace=True)
        self.train_X = model_input[["StateHoliday", "SchoolHoliday"]]
        self.train_y = model_input[["Sales"]]
        print("inside constructor on Fit")
        self.model = LinearGAM().fit(
            X=self.train_X[["StateHoliday", "SchoolHoliday"]], y=self.train_y["Sales"]
        )

    def predict(self, context, data_input):
        """
        Store Date StateHoliday SchoolHoliday - input columns by design choice
        """
        # here we are accepting the fact that our input might be a numpy array
        # keep your input ordered, Pandas DF are not like Spark and they dont work on a schema but on numbering
        model_input = pd.DataFrame(
            data_input, columns=["Store", "Horizon", "Date", "StateHoliday", "SchoolHoliday"]
        )
        # becasue it's forecasting I sort per date
        model_input.sort_values("Date", inplace=True)
        model_input["Sales_Pred"] = np.maximum(
            self.model.predict(X=model_input[["StateHoliday", "SchoolHoliday"]]), 0
        )
        return model_input[
            [
                "Date",
                "Store",
                "Horizon",
                "Sales_Pred",
            ]
        ]