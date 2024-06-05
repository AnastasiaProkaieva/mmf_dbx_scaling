# Databricks notebook source
# MAGIC
# MAGIC %md 
# MAGIC The code has been done to follow the following pattern: evoke 1 object with all pretrained models that is based on the ID of interest that contains independently pretrained forecasting models.
# MAGIC
# MAGIC
# MAGIC Step per step approach: 
# MAGIC - you train your models independently and store them under Experiment of MlFlow with their corresponding RunIDs
# MAGIC   - each experiment correspond to a FIT of your model 
# MAGIC     - we would initiate a new experiment each time we are recreating a new model(using FIT)
# MAGIC   - each run ID = 1 model
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC **Keep in mind** that if you want to do only `fit().predict()` at each evokation you need to serialize only 1 model where you include fit under the `.predict` and you dont have to serialize dictionary of artifacts since you do not require an object per model (it will be created at each fit during the serving). <br>
# MAGIC There are a few things to keep in mind with that approach: 
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

model_list.head()

# COMMAND ----------


model_list[["encoded_model"]].head(2).to_json()

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
    def __init__(self, model_list = []):
      # generate the model list - we could provide that if calling the class from outside mlflow
      self.model_list = model_list

    def load_context(self, context):
        # Get JSON dictionary from the artifacts for all models and load it as pd dataframe 
        model_list = pd.DataFrame.from_records(mlflow.artifacts.load_dict(context.artifacts['model_list']))
        print(model_list)
        model_list["model_artifact"] = [pickle.loads(urlsafe_b64decode(artifact.encode("utf-8"))) for artifact in model_list['encoded_model']]
        self.model_list = model_list.set_index('Store')

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
          print(pd_store_df.head())
          if store_id in self.model_list.index.to_list():
            model = self.model_list.loc[store_id, 'model_artifact']
            predict_df = model.predict(context=None, 
                                       data_input = pd_store_df[[
                                         "Store", "Date", 
                                         "SchoolHoliday", "Promo", 
                                         "Mean_TemperatureC"
                                         ]])
            output_df = pd.concat([output_df, predict_df]) 
          else:
            print("Model was not found, add your implementation logic for fit on the fly here")

        return output_df

# COMMAND ----------

model_list = spark.table(f"{database_name}.sales_model_table_main").toPandas() # for using the latest version
# artifact needs to be recorded as a JSON in model registry -- transforming model pandas dataframe into dict
model_list['training_date'] = [date.strftime('%Y-%m-%d') for date in model_list['training_date']]
encoded_model_noarray = [ i[0] for i in model_list['encoded_model']]
model_list["encoded_model"] = encoded_model_noarray
model_list_dict = model_list.to_dict(orient='records')
model_list = model_list.set_index('Store') # setting the index for using later below 

# COMMAND ----------


from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
# url used to send the request to your model from the serverless endpoint
db_host = dbutils.secrets.get("mlaction", "rag_sp_host")
db_token = dbutils.secrets.get("mlaction", "rag_sp_token")

os.environ['DATABRICKS_TOKEN'] = db_token
os.environ['DATABRICKS_HOST'] = db_host

model_name = f"{database_name}.model_wrapper_serving_fsa"

# recording the the model artifact in the model registry
with mlflow.start_run(run_name = "forecast_wrapper") as run:
  
  # loading the artefacts to the global model
  mlflow.log_dict(model_list_dict, 'model/artifact_array.json') # recording the artifact in the path model/
  artifacts = {'model_list' : run.info.artifact_uri + '/model/artifact_array.json'} # recording the artifact so that it's callable in the context
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
      artifacts= artifacts,
      training_set=training_set,
      registered_model_name=model_name,
      code_path = ['wrapper_model.py']
  )


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Checking our results in Batch First 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Wint & Without Feature Store 

# COMMAND ----------

logged_model = f'runs:/{run.info.run_id}/model/data/feature_store/raw_model'
model_version = "6"

main_score_df_no_features = training_set.load_df().select("Store","Date")
main_score_df_features = training_set.load_df().drop("Sales")

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

# MAGIC %md ## Serving our Final Wrapper

# COMMAND ----------

# create a serving point for the model table 
model_endpoint_name = 'dais-forecasting-main-onlineFSA-main'
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
