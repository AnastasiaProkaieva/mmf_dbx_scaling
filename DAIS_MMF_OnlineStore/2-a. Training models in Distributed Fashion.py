# Databricks notebook source
# MAGIC %md 
# MAGIC # What's new in 2024 from Databricks 🤘
# MAGIC
# MAGIC With the advance of LLM there are new features that are available and can be used for the gas forecasting and other forecasting platforms. Namely these are:
# MAGIC - online stores (low latency serving of Delta table - Features, similar to Cosmos and Dynamo)
# MAGIC - features specs (using delta tables within a model for offline and online tables, published Online Store)
# MAGIC
# MAGIC 👉 [documentation](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html)

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering==0.2.0 databricks-sdk==0.20.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1. Installing and importing required libraries 

# COMMAND ----------

# MAGIC %run ./global_code_import 

# COMMAND ----------

from aux_scripts import *
from wrapper_model import *

# setting up MLFlow clients
client = MlflowClient()

## Set MLFlow Experiment 
ct = datetime.datetime.now()
date_now_str = f"{ct.year}_{str(ct.month).zfill(2)}_{ct.day}"
# Setting our experiment to kep track of our model with MlFlow 
experiment_name = "forecasting_dais"
experiment_path = f"/Shared/{experiment_name}_rundayis_{date_now_str}"

experiment_fct = create_set_mlflow_experiment(experiment_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2. Setting up the features tables
# MAGIC
# MAGIC The training model is using features that are currently fed in the API call. In the ideal case, the features are a specific table that ensures that:
# MAGIC - Features are computed in a consistent manner (there is a consistency between offline and online features)
# MAGIC - Features are shared accross models and lineage is shown
# MAGIC - Features are looked up rather than being computed for each of the models
# MAGIC
# MAGIC Features are just regular delta tables with a primary keys - from then, we can synchronise those offline delta tables to an online store so that they can be accessed with low latency.

# COMMAND ----------

# Check that you can see the dataset and the features (offline) are pulled indeed
display(training_set.load_df())

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Part 2.2 Testing on a sample first 

# COMMAND ----------

# Testing on 1 store 
salesDF = training_set.load_df()
# splitting dataset to the Train and Test 
train_sales = salesDF.filter("Date < '2015-01-01'")
print("Amount of training points: ", train_sales.count())
test_sales = salesDF.filter("Date > '2015-01-01'")
print("Amount of testing points: ",test_sales.count())

# let's try with one store before scaling this to all thouthands stores 
store_id = 1001
store_df_train = (train_sales.filter(f"Store == {store_id}").toPandas())
store_df_test = test_sales.filter(f"Store == {store_id}").drop("Sales").toPandas()

artifact_name = f"model_custom_{store_id}"

# COMMAND ----------

model_wrapper = ForecastingModelProphet() ## Choose your model, we have tested a few models, but attaching Prophet example
with mlflow.start_run(run_name=f"testing_model_wrapper_{store_id}") as run:
    model_wrapper.fit(store_df_train)
    mlflow.pyfunc.log_model(
                artifact_path=artifact_name, 
                python_model= model_wrapper,
                )

# let's score our first model 
model_testing = mlflow.pyfunc.load_model(f'runs:/{run.info.run_id}/model_custom_1001')
model_testing.predict(store_df_test).iloc[:10,:]

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
# MAGIC For 50 models it was 4.09 minutes, I assume it would grow linearly(even it's not exactly the case), hence bringing us to 4*1215/50 ~ 100 min and what if we have 10K stores /items and more ? =>  too long :)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Part 2.3 Increase your training dataset (Optional)
# MAGIC
# MAGIC My dataset was quite small, only 1115 individual Stores but you would have way more most of the time. To demonstrate this indeed is important to distribute training I will increase the training dataset by adding GroupBy over a State and also over a new varioable call - Horizon. 
# MAGIC This will increase the dataset significantly.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part3: Creating individual forecasting models per Store

# COMMAND ----------

print("Amount of individuals models:", len(train_sales.select("Store").distinct().collect()))

# AQE is a great tool but it may cause issues when you have squeueed tasks 
spark.conf.set("spark.sql.adaptive.enabled", "false")
models_amount = len(train_sales.select("Store").distinct().collect())
#it's better to repartition your dataset before to assure 1 task per store
# makes shuffle in advance for you
df_full = train_sales.repartition(models_amount, ["Store"])
# to run repartitioning you need to call an action
df_full.count()

# COMMAND ----------

# # Repartition the dataset to minimize shuffle and remove AQE to speed up the training (AQE is made for improving data query -- not ML training)
# # Get the number of workers for fixed cluster - if you use autoscaling - which is not recommended for SLA - you need to deal with that differently
# # If the cluster is single node - then we take 1 for the driver
# number_workers = max(int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers")), 1)
# spark.conf.set("spark.sql.adaptive.enabled", False)

# # Change the shuffle partitions in the case of a cluster. For single node leave as default.
# spark.conf.set("spark.sql.shuffle.partitions", sc.defaultParallelism * number_workers) # (the tasks available = CPU x Workers) 

# logger.info(f"the parallelism used will be: {sc.defaultParallelism}")

# COMMAND ----------

from datetime import date
import pickle
import zlib
import json
from base64 import urlsafe_b64decode, urlsafe_b64encode
from wrapper_model import *

def fit_final_model_udf(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Your input DataFrame will have following columns as an input:

    "Store" -> Columns on which we are going to preform a groupBy to seperate our individual datasets
    "Date", "Sales", "SchoolHoliday", "Promo", "Mean_TemperatureC" -> Columns that will be used for Training
    "run_id", "experiment_id" -> Column that are necessary for logging under MLFlow Artifact 
    
    NOTE: 
    In a case you are using a very simple model - Linear, it does not contain parameters,
    hence you may not even care of logging the model into the mlflow for each fit.
    The only what you need to keep track is the version of the package under. 
    Nevertheless we are going to demonstrate how this can be done if you require to run a more complex model,
    e.g. Prophet, XgBoost, SKTime etc

    """
    #starting timer
    start_time = time.time()

    #from ServingForecasting import wrapper_model
    #model = wrapper_model.ForecastingModelProphet()  
    import prophet as Prophet 
    model = ForecastingModelProphet()
    horizon = model.horizon

    df_pandas = df_pandas.fillna(0).sort_values("Date").copy()
    X = df_pandas[["Store", "Date", "Sales", "SchoolHoliday", "Promo", "Mean_TemperatureC"]]
    y = df_pandas.loc[:, ["Sales"]]

    store = df_pandas["Store"].iloc[0]
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
                infer_code_paths = True, # try this one 
            )
            # logging feature version
            mlflow.log_param(f"model_trained", date.today())
            train_time = [time.time() - start_time]
            mlflow.log_param(f"train_time", train_time)
            # pay attention that the artifact you are writting is the same as your model
            artifact_uri = f"runs:/{run.info.run_id}/{artifact_name}"
           
    # we are going to encode our model 
    model_encoder = str(urlsafe_b64encode(pickle.dumps(model)).decode("utf-8"))
    # Place the pkl under the main class for simplicity 
    #model_encoder = model.pkl() 
    
    # Create a return pandas DataFrame that matches the schema above
    returnDF = pd.DataFrame(
                    [[store, artifact_uri, [model_encoder]]],
        columns = ["Store", "model_path", "encoded_model"],
        )

    return returnDF

# COMMAND ----------

# %sql
# DROP TABLE IF EXISTS ${catalog}.${schema}.sales_model_table;
# -- Create the model table that will hold all teh models and make sure it can be use as a feature (Primary Key Not Null)
# CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.sales_model_table
# (
#   Store STRING NOT NULL PRIMARY KEY COMMENT 'This is the Store reference ID',
#   encoded_model STRING COMMENT 'The model artifact cloudpicked in base64',
#   training_date DATE COMMENT 'Date at which the training was done'
# )
# COMMENT 'This table holds all the models used for Sales forecasting'
# TBLPROPERTIES (delta.enableChangeDataFeed = true);

# -- Adding tags to the table
# ALTER TABLE ${catalog}.${schema}.sales_model_table
# SET TAGS ('team'='sales', 'project'='forecast', 'type'='model_table')

# COMMAND ----------

import pyspark.sql.types as t
import pyspark.sql.functions as F 

trainReturnSchema = t.StructType([
  t.StructField("Store", t.StringType()),  # unique store ID
  t.StructField("model_path", t.StringType()), # path to the model for a given combination
  t.StructField("encoded_model", t.ArrayType(t.StringType())), # array as encoded string of the model 
])

experiment_id = experiment_fct.experiment_id

with mlflow.start_run() as run:
  run_id = run.info.run_id
  
  modelDirectoriesDF = (
            df_full
            .withColumn("run_id", F.lit(run_id))
            .withColumn("experiment_id", F.lit(experiment_id))  
            .groupby("Store")
            .applyInPandas(fit_final_model_udf, schema=trainReturnSchema)
            .withColumn("training_date", F.current_date())
      )
  

modelDirectoriesDF.select("Store", "encoded_model", "training_date").write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{database_name}.sales_model_table_main")
#modelDirectoriesDF.display()

# COMMAND ----------

# ## You can or save it back with the main Features, or keep only models 
# forecastDF = (df_full.join(modelDirectoriesDF, on=["Store"], how="left"))

# # COMMAND ----------

# def apply_model_decode(df_pandas: pd.DataFrame) -> pd.DataFrame:
#     # Get model path from metadata
#     payload = df_pandas["encoded_model"].iloc[0]
#     features = ["Store", "Horizon", "Date", "StateHoliday", "SchoolHoliday"]
#     # Subset inference set to features
#     X = df_pandas[features].sort_values("Date")
#     # Load and apply model to inference set
#     model = pickle.loads(urlsafe_b64decode(payload.encode("utf-8")))
#     #model = mlflow.pyfunc.load_model(model_path)
#     return_df = model.predict(None, X) # should return a DF 
#     return_df["Date"] = return_df["Date"].astype("str")
#     return return_df[["Store", "Horizon", "Date", "Sales_Pred"]]

# # COMMAND ----------

# predictionsDF = (forecast_df.groupBy("Store")
#                  .applyInPandas(apply_model_decode, schema="Store string, Horizon integer, Date string, Sales_Pred float"))

# COMMAND ----------

try:
  spark.sql(f"ALTER TABLE {catalog}.{schema}.sales_model_table_main ALTER COLUMN Store SET NOT NULL")
  park.sql(f"ALTER TABLE {catalog}.{schema}.sales_model_table_main ADD CONSTRAINT sales_model_table_main_pk PRIMARY KEY(Store);")
except:
  print("Yourt table already contains the restriction")

# COMMAND ----------

# import pyspark.sql.types as t
# import pyspark.sql.functions as F 

# trainReturnSchema = t.StructType([
#   t.StructField("Store", t.StringType()),  # unique store ID
#   #t.StructField("Horizon", t.IntegerType()),  # unique horizon ID
#   t.StructField("model_path", t.StringType()), # path to the model for a given combination
#   t.StructField("encoded_model", t.StringType()), # encoded string of the model 
# ])

# experiment_id = experiment_fct.experiment_id

# with mlflow.start_run() as run:
#   run_id = run.info.run_id
  
#   modelDirectoriesDF = (df_full
#                         .withColumn("run_id", F.lit(run_id))
#                         .withColumn("experiment_id", F.lit(experiment_id))  
#                         .groupby("Store")
#                         .applyInPandas(fit_final_model_udf, schema=trainReturnSchema)
#                         .withColumn("training_date", F.current_date())
#                        )
  
# ## You can or save it back with the main Features, or keep only models 
# #combinedDF = (df_full.join(modelDirectoriesDF, on=["Store"], how="left"))

# modelDirectoriesDF.select("Store", "encoded_model", "training_date").write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{database_name}.sales_model_table")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ${catalog}.${schema}.sales_model_table_main LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- since the model table is a Delta Table - we have history, time travel (rollback) and traceability of the operations
# MAGIC -- including merge overwrite etc....
# MAGIC -- we could include CDF also to this table for using triggered update of the online table
# MAGIC DESCRIBE HISTORY ${catalog}.${schema}.sales_model_table_main LIMIT 10

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Verifying it works for 1 store 
# MAGIC

# COMMAND ----------


store_id = '1001'

# Get the model dataframe from the model table
model_list = spark.table(f'{database_name}.sales_model_table_main').filter(f"Store = '{store_id}'").toPandas()

if not model_list.empty:
  model_list.set_index('Store', inplace=True)
  # Load the model artifact from the model table
  payload = model_list.loc[store_id, "encoded_model"][0] # here we are using encoded_model as array of strings                          
  model_test = pickle.loads(urlsafe_b64decode(payload.encode("utf-8")))

  # Load the data to be used for prediction
  data_mocking_test = (
    test_sales.filter(f"Store = '{store_id}'").drop("Sales").toPandas()
    )

  # Launch the predict
  start_time = time.time()
  predict_df = model_test.predict(context=None, data_input=data_mocking_test)
  predict_time = time.time()
  #logger.info(f"Predict ran in {predict_time - start_time}s")
  predict_df.display()
else:
  #logger.info(f"no models found for that Store") 
  print(f"no models found for that Store") 

# COMMAND ----------


