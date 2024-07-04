import os
import pickle
import numpy as np
import pandas as pd
import mlflow

from prophet import Prophet
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

    def __init__(self):
        #super().__init__()
        self.main_features = ["Store", "Date", "Sales"]
        self.features = ["SchoolHoliday", "Promo", "Mean_TemperatureC"]
        self.horizon = 30
    
    def pkl(self):
      # encoding the model in string so that it can be serialized in the artficat dictionary
      # could use zlib to further compress ~x2 the model to speed-up download/upload to s3
      return base64.b64encode(cloudpickle.dumps(self)).decode('utf-8')

    def fit(self, data_input):
        """
        Store Date Sales StateHoliday SchoolHoliday - input columns by design choice
        """
        columns = self.main_features+self.features
        model_input = data_input.fillna(0)
        # becasue it's forecasting I sort per date
        model_input.sort_values("Date", inplace=True, ascending=True)
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
        if len(self.features)>0:
          _ = [self.model.add_regressor(feature_name) for feature_name in self.features]
        self.model.fit(model_input)

    def predict(self, context, data_input):
        """
        Store Date StateHoliday SchoolHoliday - input columns by design choice
        """
        model_input = data_input.fillna(0)
        model_input.sort_values("Date", inplace=True, ascending=True)
        model_input.rename(columns={"Date": "ds"}, inplace=True)
  
        future = self.model.make_future_dataframe(periods=self.horizon, 
                                        freq='d', 
                                        include_history=False)
  
        model_input['ds'] = pd.to_datetime(model_input['ds'])
        future = future.merge(model_input[['ds']+self.features], on='ds', how='left').fillna(0)
        forecast = self.model.predict(future)
        forecast.rename(columns={"yhat": "Sales_Pred", "ds": "Date"}, inplace=True)
        forecast["Store"] = model_input["Store"].iloc[0]
       
 
        return forecast[
            [
                "Date",
                "Store",
                "Sales_Pred",
            ]
        ]