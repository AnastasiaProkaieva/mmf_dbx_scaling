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

# COMMAND ----------

# DBTITLE 1,Lanching some prep behind the scene
# MAGIC %run ./global_code_import

# COMMAND ----------

# url used to send the request to your model from the serverless endpoint
db_host = dbutils.secrets.get("mlaction", "rag_sp_host")
db_token = dbutils.secrets.get("mlaction", "rag_sp_token")

os.environ['DATABRICKS_TOKEN'] = db_token
os.environ['DATABRICKS_HOST'] = db_host

# COMMAND ----------


# Creating our Online Tables to use this table in RealTime Feature Join 
create_online_table(f"{database_name}.sales_model_table_main", ["Store"], timeseries_key=None)
create_online_table(f"{database_name}.extra_sales_weather_fs", ["Store", "Date"], timeseries_key=None)
create_online_table(f"{database_name}.main_features_sales_fs", ["Store", "Date"], timeseries_key=None)

# COMMAND ----------

fe = FeatureEngineeringClient()
# Creating a Spec to be served outside as a model serving call, this one is optional 
create_feature_spec(f"{database_name}.sales_model_table_main", f"{database_name}.sales_models_table_feature_spec", "Store")

# COMMAND ----------

# create a serving point for the model table 
model_table_endpoint_name = 'dais-forecast-serving-spec'
create_fsm_serving(model_table_endpoint_name, f"{database_name}.sales_models_table_feature_spec")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Verifying that our Feature endpoint works

# COMMAND ----------


# Checking that you can pull a model from the table
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get('browserHostName').get()
# Generate the url for the Feature Endpoint
url = f"https://{workspace_url}/serving-endpoints/{model_table_endpoint_name}/invocations"
# Generate the temporary token
databricks_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token
headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}

data = {"dataframe_records": [{"Store": "1001"}]}
data_json = json.dumps(data, allow_nan=True)

response = requests.request(method='POST', headers=headers, url=url, data=data_json)
if response.status_code != 200:
  raise Exception(f'Request failed with status {response.status_code}\n{response.text}')

model_example = response.json()["outputs"][0].get('encoded_model')
pprint(pickle.loads(urlsafe_b64decode(model_example[0].encode("utf-8"))))

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


