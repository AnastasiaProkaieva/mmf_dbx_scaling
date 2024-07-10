# Databricks notebook source
# MAGIC %md 
# MAGIC ## Example 3.2
# MAGIC   

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering==0.2.0 databricks-sdk==0.20.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------


from aux_scripts import *
from wrapper_model import *

# COMMAND ----------

# MAGIC %run ./global_code_import

# COMMAND ----------

# Custom implementation of a Python ordered dictionary with a size limit. Key-value pairs are stored in a dictionary object. 
# When a key is accessed (either to set or get a value), it is moved to the head of the dictionary.
# If this size of the dictionary exceeds the maximum size limit, the least recently used key-value pair is removed to accommodate the new entry.

class LimitedOrderedDict:
    def __init__(self, max_size=100, data = {}):
        self.data = OrderedDict(data)
        self.max_size = max_size

    # Set a key value pair in the dictionary.
    # If the key already exists in the dictionary, it is moved to the head or beginning, thereby making it the most recent.
    # If the size of the dictionary exceeds the maximum size, the least recently used key value pair is removed to accommodate the new entry.
    def __setitem__(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        else:
            if len(self.data) >= self.max_size:
                self.data.popitem(last=False)
            self.data[key] = value

    # Get the value associated with the provided key, move the key value to the head of the dictionary or raise an exception.
    def __getitem__(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        else:
            raise KeyError(f"Key '{str(key)}' not found")

    # Get a dictionary with the keys and values stored in the object.
    def get_ordered_dict(self):
        return self.data
    
    # Get the value associated with the provided key after moving the key value to the head of the dictionary 
    # or return the default value if the key is not present.
    # This does not raise an exception when the key is not present.
    def get(self, key, default=None):
        if key in self.data:
            self.data.move_to_end(key)
        return self.data.get(key, default)

    # Return a string representation of the dictionary with keys and values
    def __repr__(self):
        return repr(self.data)
      

# COMMAND ----------

from datetime import date
import pickle
import zlib
import json
from base64 import urlsafe_b64decode, urlsafe_b64encode


# even if you're writing Pyfunc code in a notebook, note that notebook state is *not* copied into the model's context automatically.
# as demonstrated below, state must be passed in explicitly through artifacts and referenced via the context object.
class MultiModelPyfunc(mlflow.pyfunc.PythonModel):
    """
    the multi model will load the artifact which contains all models from the Delta sales_model_table
    this artifact is loaded in memory and served as a pandas dataframe
    the input dataframe is then split per Store and the predict is done per Store
    if a model does not exist we call a fit function for that dataset - for that we need the Sales label as an input - this is done through the Sales feature table
    """
    def __init__(self):
      super().__init__()

    def load_context(self, context):
        pass
    
    ## An example if you need to add a data processing on the fly to your data 
    def process_input(self, raw_input):
        pass 

    def predict(self, context, model_input):
        """
        expecting the input to be a dataframe from model serving as an input
        the dataframe contains different Store and is enriched with features: Weather, Promos, or manual input for Sales
        """        
        output_df = pd.DataFrame()
        for store_id, pd_store_df in model_input.groupby('Store'):
          # getting the model from pd_store_df
          model_pickled = pd_store_df.iloc[0,:]["encoded_model"][0]
          model = pickle.loads(urlsafe_b64decode(model_pickled.encode("utf-8")))
          predict_df = model.predict(context=None, 
                                       data_input = pd_store_df[[
                                         "Store", "Date", 
                                         "SchoolHoliday", "Promo", 
                                         "Mean_TemperatureC"
                                         ]])
          output_df = pd.concat([output_df, predict_df], ignore_index=True)

        return output_df

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
# url used to send the request to your model from the serverless endpoint
db_host = dbutils.secrets.get("dais_mmf", "sp_host")
db_token = dbutils.secrets.get("dais_mmf", "sp_token")

os.environ['DATABRICKS_TOKEN'] = db_token
os.environ['DATABRICKS_HOST'] = db_host

model_name = f"{database_name}.model_wrapper_serving_fsm"

# recording the the model artifact in the model registry
with mlflow.start_run(run_name = "forecast_wrapper") as run:

  # since mlflow cannot infer the requirement from the libraries used for training -- the class being custom -- we need to specify the libraries
  reqs = mlflow.pyfunc.get_default_pip_requirements() + [
    "mlflow==" + mlflow.__version__, 
    "pandas==" + pd.__version__,
    "scipy==" + scipy.__version__,
    "databricks-feature-lookup==0.*",
    "prophet=="+prophet.__version__,
    ]
  
  # logging the model using the feature store API 
  fe.log_model(
      artifact_path = "model",
      model = MultiModelPyfunc(),
      flavor= mlflow.pyfunc,
      pip_requirements= reqs,
      training_set=features_set,
      registered_model_name=model_name,
      code_path = ['wrapper_model.py']
  )


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Checking our results in Batch First 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### With & Without Feature Store 

# COMMAND ----------

logged_model = f'runs:/{run.info.run_id}/model/data/feature_store/raw_model'
model_version = "3"

main_score_df_no_features = features_set.load_df().select("Store","Date")
main_score_df_features = features_set.load_df().drop("Sales")

# let's try with one store before scaling this to all thouthands stores 
store_id = "1"
test_sales_features = main_score_df_features.filter(f"Store == {store_id}")
test_sales_no_features = main_score_df_no_features.filter(f"Store == {store_id}")

print("Testing the predict with Features")
data_mocking_test = test_sales_features.toPandas()
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
predict_df = loaded_model.predict(data_mocking_test)
predict_df.head()

print("Testing the predict without features using FS ")
predict_df = fe.score_batch(
  model_uri=f'models:/{model_name}/{model_version}',
  # it's a Spark UDF so we need to confort the output format 30 days 
  df=test_sales_no_features.filter(
    (f.col("Date") >= "2015-01-01") & 
    (f.col("Date") < "2015-01-31"))
)
predict_df.display()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Serving our final wrapper

# COMMAND ----------

# create a serving point for the model table 
model_endpoint_name = 'dais-forecasting-main-onlineFS-main'
create_fsm_serving(model_endpoint_name, model_name, version = model_version)

# COMMAND ----------

testing_serving_df = test_sales_no_features.filter(
    (f.col("Date") >= "2015-01-01") & 
    (f.col("Date") < "2015-01-31")).toPandas()

testing_serving_df['Date'] = testing_serving_df['Date'].astype('str')
data_mocking_short = testing_serving_df.to_dict(orient="split")
ds_dict = json.dumps({"dataframe_split": data_mocking_short}) # can pass cls = DateTimeEncoder if do not do str conversion fro your input 
ds_dict

# COMMAND ----------

response = evoke_serving_endpoint(model_endpoint_name, ds_dict)
pd.DataFrame(response.json()['predictions'] )

# COMMAND ----------


