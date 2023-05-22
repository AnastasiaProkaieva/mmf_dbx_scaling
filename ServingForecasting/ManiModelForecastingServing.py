# Databricks notebook source
# DBTITLE 0,kirjjbhgthkbdlkrnjulvulergujbgljdnb
# MAGIC %md 
# MAGIC # Description 
# MAGIC This is an example use case how to forecast and serve multiple independent models under 1 endpoint on Databricks Serving Endpoints. 
# MAGIC For this example I am going to use Rossman Drugstore Dataset, the data was taken from the Kaggle Page [HERE]('').
# MAGIC If you would like to know how to get your data directly from kaggle page using thier API and ingest it into Delta Table on Databricks  check the notebook attached to it `get_data`.
# MAGIC
# MAGIC In order to run the notebook on Databricks you require the following:
# MAGIC - DBR ML 12.2 with multi node (my cluster configuration is 4 nodes 8 CPU each)

# COMMAND ----------

display(spark.read.table("hive_metastore.ap.hack_ap_rossmann_time_series"))

# COMMAND ----------

salesDF = (spark.read.table("hive_metastore.ap.hack_ap_rossmann_blog")
                .dropDuplicates()# dropping duplicates if any
                .select("Store", "Date", "Sales-1", "StateHoliday", "SchoolHoliday")# selecting columns of interest 
                .withColumnRenamed("Sales-1", "Sales")# renaming a column 
          )
          
display(salesDF)

# COMMAND ----------

display(salesDF.filter("Store = 1001"))

# COMMAND ----------

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from ServingForecasting.wrapper_model import *
from ServingForecasting.aux_scripts import *

client = MlflowClient()

ct = datetime.datetime.now()
date_now_str = f"{ct.year}_{str(ct.month).zfill(2)}_{ct.day}"
# Setting our experiment to kep track of our model with MlFlow 
experiment_name = "forecasting_ap"
experiment_path = f"/Shared/{experiment_name}_rundayis_{date_now_str}"

experiment_fct = create_set_mlflow_experiment(experiment_path)

# COMMAND ----------

# splitting dataset to the Train and Test 
train_sales = salesDF.filter("Date < '2015-01-01'")
print("Amount of training points: ", train_sales.count())
test_sales = salesDF.filter("Date > '2015-01-01'")
print("Amount of testing points: ",test_sales.count())

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Training our first model 

# COMMAND ----------

 # let's try with one store before scaling this to all thouthands stores 
store_id = 1001
store_df_train = train_sales.filter(f"Store == {store_id}").toPandas()
store_df_test = test_sales.filter(f"Store == {store_id}").drop("Sales").toPandas()

model_wrapper = ForecastingModelProphet()

artifact_name = f"model_custom_{store_id}"
 with mlflow.start_run(run_name=f"testing_model_wrapper_{store_id}") as run:
    model_wrapper.fit(store_df_train)
    mlflow.pyfunc.log_model(
                artifact_path=artifact_name, 
                python_model= model_wrapper,
                #input_example = store_df_train.iloc[:10,:]
                )

# let's score our first model 
model_testing = mlflow.pyfunc.load_model(f'runs:/{run.info.run_id}/model_custom_1001')
model_testing.predict(store_df_test)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 
# MAGIC ### Benchmarking
# MAGIC
# MAGIC ```
# MAGIC import time 
# MAGIC def benchmark_calc(store_id):
# MAGIC     time.sleep(5)
# MAGIC     return 1
# MAGIC
# MAGIC predictions_dict = {f"{store_id}": benchmark_calc(store_id) for store_id in range(1,100,1)}
# MAGIC ```
# MAGIC
# MAGIC For 50 models it was 4.09 minutes, I assume it would grow linearly, hence bringing us to 4*1215/50 =>  too long :) 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part2: Creating individual forecasting models per Store

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Ingesting a Horizon Notion 
# MAGIC
# MAGIC There are a lot of models such a Prophet that requires you to provide the forecasting_Horizon. 
# MAGIC
# MAGIC For this example notebook, I will create a Horizon column just to synthetically increase the size of the dataset and to show you how simply you can achieve results by using Spark when the complexity of your forecasting colution is growing. 

# COMMAND ----------

from ServingForecasting import aux_scripts as scripts
train_horizons = scripts.add_horizons(train_sales, horizons=7, return_df=True)

# COMMAND ----------

display(train_horizons)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating Main Forecating Fit Func 

# COMMAND ----------

from datetime import date
import pickle
import json
from base64 import urlsafe_b64decode, urlsafe_b64encode


def fit_final_model_udf(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Your input DataFrame will have following columns as an input:

    "Store", "State" -> Columns on which we are going to preform a groupBy to seperate our individual datasets
    "Date", "Sales", "StateHoliday", "SchoolHoliday" -> Columns that will be used for Training
    "run_id", "experiment_id" -> Column that are necessary for logging under MLFlow Artifact 
    
    NOTE: 
    In a case you are using a very simple model - Linear, it does not contain parameters,
    hence you may not even care of logging the model into the mlflow for each fit.
    The only what you need to keep track is the version of the package under. 
    Nevertheless we are going to demonstrate how this can be done if you require to run a more complex model,
    e.g. Prophet, XgBoost, SKTime etc

    """
    from ServingForecasting import wrapper_model
    import prophet as Prophet 
    model = wrapper_model.ForecastingModelProphet()    

    df_pandas = df_pandas.fillna(0).sort_values("Date").copy()
    X = df_pandas[["Store", "Horizon", "Date", "Sales", "StateHoliday", "SchoolHoliday"]]
    y = df_pandas.loc[:, ["Sales"]]

    store = df_pandas["Store"].iloc[0]
    horizon = df_pandas["Horizon"].iloc[0]
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0]  # Pulls run ID to do a nested run
    experiment_id = df_pandas["experiment_id"].iloc[0]
    artifact_name = f"model_custom_{store}_{horizon}"

    # Resume the top-level training
    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as outer_run:
        #
        # Create a nested run for the specific device
        with mlflow.start_run(
            run_name=f"store_{store}_{horizon}", nested=True, experiment_id=experiment_id
        ) as run:
            #Defining Model Pipeline here
            model.fit(X)
            
            mlflow.pyfunc.log_model(
                artifact_path=artifact_name,
                python_model=model,
            )
            # logging feature version
            mlflow.log_param(f"model_trained", date.today())
            # pay attention that the artifact you are writting is the same as your model
            artifact_uri = f"runs:/{run.info.run_id}/{artifact_name}"
           
    # we are going to encode our model 
    model_encoder = str(urlsafe_b64encode(pickle.dumps(model)).decode("utf-8"))

    # Create a return pandas DataFrame that matches the schema above
    returnDF = pd.DataFrame(
        [[store, horizon, n_used, artifact_uri, model_encoder]],
        columns=["Store", "Horizon", "n_used", "model_path", "encode_model"],)

    return returnDF

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Repartition our dataset 

# COMMAND ----------

print("Amount of individuals models:", len(train_horizons.select("Store","Horizon").distinct().collect()))

# COMMAND ----------

# AQE is a great tool but it may cause issues when you have squeueed tasks 
spark.conf.set("spark.sql.adaptive.enabled", "false")
models_amount = len(train_horizons.select("Store","Horizon").distinct().collect())
#it's better to repartition your dataset before to assure 1 task per store
# makes shuffle in advance for you
df_full = train_horizons.repartition(models_amount, ["Store", "Horizon"])
# to run repartitioning you need to call an action
df_full.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Calling our ApplyInPandas

# COMMAND ----------

import pyspark.sql.types as t
import pyspark.sql.functions as F 

trainReturnSchema = t.StructType([
  t.StructField("Store", t.StringType()),  # unique store ID
  t.StructField("Horizon", t.IntegerType()),  # unique horizon ID
  t.StructField("n_used", t.IntegerType()),    # number of records used in training
  t.StructField("model_path", t.StringType()), # path to the model for a given combination
  t.StructField("encode_model", t.StringType()), # encoded string of the model 
])

experiment_id = experiment_fct.experiment_id

with mlflow.start_run() as run:
  run_id = run.info.run_id
  
  modelDirectoriesDF = (df_full
                        .withColumn("run_id", F.lit(run_id)) # Add run_id
                        .withColumn("experiment_id", F.lit(experiment_id))  
                        .groupby("Store", "Horizon")
                        .applyInPandas(fit_final_model_udf, schema=trainReturnSchema)
                        .cache()
                       )
  
combinedDF = (df_full.join(modelDirectoriesDF, on=["Store", "Horizon"], how="left"))

display(combinedDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Saving results under Delta Table

# COMMAND ----------

combinedDF.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable("hive_metastore.ap.hack_ap_rossmann_blog_predictions")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part 3: Using Forecasing Model to make predictions

# COMMAND ----------

# Reading Back forecasted objects 
forecast_df = spark.read.table("hive_metastore.ap.hack_ap_rossmann_blog_predictions")
display(forecast_df)

# COMMAND ----------

def apply_model_decode(df_pandas: pd.DataFrame) -> pd.DataFrame:
    # Get model path from metadata
    payload = df_pandas["encode_model"].iloc[0]
    features = ["Store", "Horizon", "Date", "StateHoliday", "SchoolHoliday"]
    # Subset inference set to features
    X = df_pandas[features].sort_values("Date")
    # Load and apply model to inference set
    model = pickle.loads(urlsafe_b64decode(payload.encode("utf-8")))
    #model = mlflow.pyfunc.load_model(model_path)
    return_df = model.predict(None, X) # should return a DF 
    return_df["Date"] = return_df["Date"].astype("str")
    return return_df[["Store", "Horizon", "Date", "Sales_Pred"]]

# COMMAND ----------

predictionsDF = (forecast_df.groupBy("Store")
                 .applyInPandas(apply_model_decode, schema="Store string, Horizon integer, Date string, Sales_Pred float"))
display(predictionsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC All the examples above are great when the amount of models stays reasonable to put it into the artifacts one by one. 
# MAGIC In this case I consider that my model is not going to change (it has no paramaters, and same model with the same package version is going to be used), I will not consider individual models for tracking and registry, while I will register and track MainWrapper model that will be served under the Databricks Endpoint. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating our WrappingModel for Serving 

# COMMAND ----------

# even if you're writing Pyfunc code in a notebook, note that notebook state is *not* copied into the model's context automatically.
# as demonstrated below, state must be passed in explicitly through artifacts and referenced via the context object.
class MultiModelPyfunc(mlflow.pyfunc.PythonModel):
    """
    ids_store (list, str) ::
      id's that correponds to keys associated to the artifacts dict
      keys will be used to withdraw object that correpond to a particular model
      {id_name}: model_object_per_id
    """

    def __init__(self):
        super().__init__()
        
    def load_context(self, context):
        # Get Dictionary with your artifacts for all models here
        json_load = context.artifacts["models_encoded"]
        with open(json_load) as json_file:
            data = json.load(json_file)
        self.models_context_dict = data

    ## An example yif you need to add a data processing on the fly to your data
    def process_input(self, raw_input):
        pass

    def select_model(self, model_input):
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Sample model requires Dataframe inputs")
        #locale_id = model_input["Store"].iloc[0]  # getting the model id from the Store name
        locale_id = f'{model_input["Store"].iloc[0]}_{model_input["Horizon"].iloc[0]}'  # getting the model id from the Store name
        return locale_id

    def predict(self, context, raw_input):
        """
        expecting the input to be in array
        """
        # we are checking if this is a PD.DF because this is the output from the Serving API into the Predict
        if type(raw_input) is type(pd.DataFrame()):
            raw_input = eval(raw_input.values[0][0])

        # we are here creating a DF for an expected income
        # if you want to retrain the model - place this under Try and add another one on except side
        model_input = pd.DataFrame(
            raw_input, columns=["Store", "Horizon", "Date", "StateHoliday", "SchoolHoliday"]
        )
        model_input["StateHoliday"] = model_input["StateHoliday"].astype("bool")
        model_input["SchoolHoliday"] = model_input["SchoolHoliday"].astype("bool")
        model_input.sort_values("Date", inplace=True)

        try:
            selected_store = self.select_model(model_input)
            print(f"Selected model {selected_store}")
            # TO DO
            # here add a part so that a model would be loaded and kep in a memory if it was already called
            # MOCK example
            # if str(selected_store) not in self.models
            #     self.models[str(selected_store)] = pickle.loads(urlsafe_b64decode(model_context.encode("utf-8")))
            # models = self.models[str(selected_store)]
            
            model_context = self.models_context_dict[str(selected_store)]
            model = pickle.loads(urlsafe_b64decode(model_context.encode("utf-8")))
            return model.predict(None, model_input)
        except:
            # here you can fit your model, add an aassert with corresponding amount of columns
            # pay attention your columns will not be the same as for Predict
            return f"This ID was not yet pre-trained, you input is {raw_input}, with a type {type(raw_input)}"

# COMMAND ----------

# %sh
# # make sure you have your folder created where you store your json with all artifacts  
# mkdir /dbfs/tmp/ap/

# COMMAND ----------

# getting files from the artifact 
# we are going to place the dict into the artifact and will extract objects of the model form the dict per store 
df_stores_models = forecast_df.select("Store", "Horizon", "encode_model").distinct().toPandas()
new_column = [f'{df_stores_models["Store"].iloc[il]}_{str(df_stores_models["Horizon"].iloc[il])}' for il in range(len(df_stores_models["Store"]))]
df_stores_models["Store_Horizon"] = new_column
# create key pair value Store_Horizon : model_input 
# if have a more complex index - need to create a composite Store_Horizon index 
# need to convert dict or to the file or to the JSON as a string to pass under artifacts 
dict_stores_models = dict(df_stores_models[["Store_Horizon","encode_model"]].values) # check if this work 

import json 
with open(f'/dbfs/tmp/ap/json_data_artifact_date{date_now_str}.json', 'w') as outfile:
    json.dump(dict_stores_models, outfile)

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

client = MlflowClient()

model_serving_name = "multimodel-serving-fct-custom-wrapper"

with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(
        "augmented-fct-model-custom",
        python_model=MultiModelPyfunc(),
        artifacts={
            "models_encoded": f"/dbfs/tmp/ap/json_data_artifact_date{date_now_str}.json"
        },
    )
    print("Your Run ID is: ", run.info.run_id)

    mv = mlflow.register_model(
        f"runs:/{run.info.run_id}/augmented-fct-model-custom", f"{model_serving_name}"
    )
    client.transition_model_version_stage(
        f"{model_serving_name}",
        mv.version,
        "Production",
        archive_existing_versions=True,
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Testing our Wrapper

# COMMAND ----------

# Select id's per store and horizon
store_id = 339
horizon_id = 1
store_df_test = (
    forecast_df
    .filter(f"Store == {store_id}")
    .filter(f"Horizon == {horizon_id}")
    .drop("Sales","n_used","encode_model","model_path")
    .toPandas()
                )

# load and score wrapped model 
try:
    model = mlflow.pyfunc.load_model(f'runs:/{run.info.run_id}/augmented-fct-model-custom')
except:
    #adding here a loaf
    model = mlflow.pyfunc.load_model('runs:/afbcd5b196a44f8f9bf70b104f872a85/augmented-fct-model-custom')

#mlflow.pyfunc.load_model(f'runs:/{run.info.run_id}/augmented-fct-model-custom')
model.predict(store_df_test.values)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part 4: Serving with Serving Endpoints

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Serialising our input 
# MAGIC By default Serving Endpoints will treat your data row by row approach. Because we are working on the forecasting use case, we require to pass a multidimensional input into the endpoint. 

# COMMAND ----------

import json 
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
 

# Adding a parsing functionfrom datetime to a  string 
# in real time we will pass a string but our input data as of now is a datetime object
store_df_test["Date"] = store_df_test["Date"].astype("str")
# keep attention that you require a numpy array for data_to_encode
json_dump = json.dumps(store_df_test.values, cls=NumpyEncoder) 
 
ds_dict_serving = {
  "dataframe_split": {
    "index":[0],# if you want more here will we [0,1,2] each index corresponds to a DF 
    "columns" : ["input_data"], 
    "data":  [json_dump]
    }
}

# COMMAND ----------

## for testing we are simulating the input shape 
data_json = json.dumps(dds_dict_serving, allow_nan=True)
model.predict(pd.DataFrame(eval(data_json)["dataframe_split"]['data']))

# COMMAND ----------

## To genrate a PAT token you should have a permission - ask your admin to generate you one 
# token = "PLACE YOUR TOKEN HERE"
## Example of a Invocation link(taken from the UI or API call for Serving Enablement)
## https://YOUR_DATABRICKS_WORKSPACE_DOMAIN/serving-endpoints/YOUR_ENDPOINT_NAME/invocations
# url = "PLACE YOUR MODEL INVOCATION LINK HERE"


def score_model(ds_dict, token, url):
  headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
  ds_dict_testing = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=ds_dict)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

score_model(ds_dict_testing, token, "model_serving_point")

# COMMAND ----------


