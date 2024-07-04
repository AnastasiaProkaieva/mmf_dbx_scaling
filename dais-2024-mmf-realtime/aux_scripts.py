# python libraries
import numpy as np
import pandas as pd
from pprint import pprint
from datetime import datetime, date
from typing import Iterator
import seaborn as sns
from collections import OrderedDict
import time, json, cloudpickle, base64, pickle, scipy, uuid, logging, os, requests, datetime
from base64 import urlsafe_b64decode, urlsafe_b64encode
import prophet

# mlflow libraries
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.deployments import get_deploy_client

# these are new libraries required for online store
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities.feature_lookup import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction
from databricks.feature_engineering.entities.feature_serving_endpoint import (
    EndpointCoreConfig,
    ServedEntity
)

# spark libraries
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType, DoubleType, BooleanType, ArrayType, FloatType, BinaryType, TimestampType
from pyspark.sql.functions import pandas_udf
import pyspark.sql.functions as f
from pyspark.sql.window import Window

# Databricks Client 
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as c
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
fe = FeatureEngineeringClient()


def create_set_mlflow_experiment(experiment_path):
  try:
    print(f"Setting our existing experiment {experiment_path}")
    mlflow.set_experiment(experiment_path)
    experiment = mlflow.get_experiment_by_name(experiment_path)
    return experiment
  except:
    print("Creating a new experiment and setting it")
    experiment = mlflow.create_experiment(name = experiment_path)
    mlflow.set_experiment(experiment_path)
    return experiment

def create_online_table(table_name, pks, timeseries_key=None, delete_old=None, spec_name="sales_models_table_feature_spec"):
    w = WorkspaceClient(token=os.environ['DATABRICKS_TOKEN'], host=os.environ['DATABRICKS_HOST'])
    
    if delete_old:
      fe.delete_feature_spec(name=f"{database_name}.{spec_name}")

    online_table_name = table_name+"_online"
    print(f"Creating online table for {online_table_name}...")
    try:
      spark.sql(f'ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')
    except:
      print("Table is already CDF enabled")
    try:
      spec = c.OnlineTableSpec(
          source_table_full_name=table_name, 
          primary_key_columns=pks, 
          run_triggered={'triggered': 'true'}, 
          timeseries_key=timeseries_key
          )
      w.online_tables.create(name=online_table_name, spec=spec)
    except Exception as e:
      print(f"there is an issue in creating the feature spec: {e.args[0]['error_code']}") 

def create_feature_spec(fs_table_name, fs_spec_name, lookup_key_name):  
  # Create a feature spec that allows querying the table from a serving endpoint [separate from the model serving]
  try:
    fe.create_feature_spec(
      name=f"{fs_spec_name}",
      features=[
        FeatureLookup(
          table_name=f"{fs_table_name}",
          lookup_key=f"{lookup_key_name}"
        )
      ]
    )
    print(f"feature spec created : {fs_spec_name}")  

  except Exception as e:
    print(f"there is an issue in creating the feature spec: {e.args[0]['error_code']}") 

def create_fsm_serving(model_endpoint_name, name, version = None):
  w = WorkspaceClient(token=os.environ['DATABRICKS_TOKEN'], host=os.environ['DATABRICKS_HOST'])

  try:
    w.serving_endpoints.create_and_wait(
        name=model_endpoint_name,
        config=EndpointCoreConfigInput(
          served_entities=[
            ServedEntityInput(
              entity_name = name,
              scale_to_zero_enabled = True,
              entity_version = version, 
              workload_size="Small",
              environment_vars ={
                      "DATABRICKS_TOKEN": "{{secrets/mlaction/rag_sp_token}}",  # <scope>/<secret> that contains an access token
                      "DATABRICKS_HOST": "{{secrets/mlaction/rag_sp_host}}", 
                  }
            )
          ]
        )
      )
    print(f"endpoint serving for {name} creating : {model_endpoint_name}")  

  except Exception as e:
    print(f"there is an issue in creating the endpoint: {e}") 


def evoke_serving_endpoint(model_endpoint_name, ds_dict): 
  db_token = os.environ.get('DATABRICKS_TOKEN')
  db_host = os.environ.get('DATABRICKS_HOST')
  # Generate the url for the Feature Endpoint
  url = f"{db_host}/serving-endpoints/{model_endpoint_name}/invocations"
  headers = {'Authorization': f'Bearer {db_token}', 'Content-Type': 'application/json'}

  response = requests.request(method='POST', headers=headers, url=url, data=ds_dict )
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}\n{response.text}')
  return response 

def prepare_mock_dataset(df): 
    df = df.drop(columns=["Sales"])
    df["Store"] = df["Store"].astype("str")
    df["Date"] = df["Date"].astype("str") 
    return df.values  


def add_horizons(spark_df, horizons=None,
                 write=None, return_df=None,
                 table2save="catalog.schema.forecasting_horizons"):
    
  if not horizons:
    horizons = 1
  # We are creating an array of 14 horizons
  horizons = [it for it in range(1, horizons+1)] # adding one because python does -1
  # lit this array over the Date column
  spark_df_H = spark_df.withColumn('Horizons', f.array([f.lit(x) for x in horizons]))
  # exploding the horizon column per Date
  spark_df_H = spark_df_H.select(
                                    f.col("Date"),
                                    f.col("Store"),
                                    f.col("Sales"),
                                    f.col("StateHoliday"),
                                    f.col("SchoolHoliday"),
                                    f.explode("Horizons").alias("Horizon")
                                )
  # call count before writting to prepare the plan 
  print("Generated Size is",spark_df_H.count())
  print("Original size is ",spark_df.count())
  
  if write:
    # append rather then overwrite the whole table when new data arrives 
    spark_df_H.write.mode("overwrite").saveAsTable(table2save)
  if return_df:
    return spark_df_H

