import os
import requests
import numpy as np
import pandas as pd
import json
import mlflow
import datetime
import pyspark.sql.functions as f

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


def prepare_mock_dataset(df): 
    df = df.drop(columns=["Sales"])
    df["Store"] = df["Store"].astype("str")
    df["Date"] = df["Date"].astype("str") 
    return df.values  


def add_horizons(spark_df, horizons=None,
                 write=None, return_df=None,
                 table2save="hive_metastore.ap.blog_forecasting_horizons"):
    
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

