import os
import pickle
import numpy as np
import pandas as pd
import mlflow

from prophet import Prophet

class ForecastingModelProphet(mlflow.pyfunc.PythonModel):
    """
    ForecastingModel class here is to fit and predict individual forecasting model
    """
    import prophet as Prophet

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
            columns=["Store", "Date", "Sales", "StateHoliday", "SchoolHoliday", "Horizon"],
        ).fillna(0)
        # becasue it's forecasting I sort per date
        model_input.sort_values("Date", inplace=True)
        model_input.rename(columns={"Sales": "y", "Date": "ds"}, inplace=True)
        
        #training model 
        self.model = Prophet(
            interval_width=0.95,
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
            ) 
        self.model.add_regressor('StateHoliday')
        self.model.add_regressor('SchoolHoliday')
        self.model.fit(model_input)

    def predict(self, context, data_input):
        """
        Store Date StateHoliday SchoolHoliday - input columns by design choice
        """
        print("Entering PREDICT ")
        # here we are accepting the fact that our input might be a numpy array
        # keep your input ordered, Pandas DF are not like Spark and they dont work on a schema but on numbering
        model_input = pd.DataFrame(
            data_input, columns=["Store", "Date", "StateHoliday", "SchoolHoliday", "Horizon"])
        model_input.sort_values("Date", inplace=True)
        model_input.rename(columns={"Date": "ds"}, inplace=True)
        try:
            horizon = model_input["Horizon"].iloc[0]
            period = 30*horizon
        except:
            horizon = 1
            period = 30
            
        future = self.model.make_future_dataframe(periods=30, 
                                        freq='d', 
                                        include_history=False)
  
        model_input['ds'] = pd.to_datetime(model_input['ds'])
        future = future.merge(model_input[['ds', 'StateHoliday','SchoolHoliday']], on='ds', how='left').fillna(0)
        forecast = self.model.predict(future)
        forecast.rename(columns={"yhat": "Sales_Pred", "ds": "Date"}, inplace=True)
        forecast["Horizon"] = horizon
        forecast["Store"] = model_input["Store"].iloc[0]
 
        return forecast[
            [
                "Date",
                "Store",
                "Horizon",
                "Sales_Pred",
            ]
        ]