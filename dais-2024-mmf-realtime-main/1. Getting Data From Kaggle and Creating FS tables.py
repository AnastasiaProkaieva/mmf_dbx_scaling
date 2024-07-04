# Databricks notebook source
# MAGIC %md 
# MAGIC ## Get your data
# MAGIC
# MAGIC We are using a dataset from a known forecasting competition on Kaggle https://www.kaggle.com/competitions/rossmann-store-sales with it's extended version. 
# MAGIC
# MAGIC
# MAGIC You will learn: 
# MAGIC - ingesting data into Delta tables 
# MAGIC - creating a Feature Store Tables 

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

dbutils.widgets.text("catalog", "ap")
dbutils.widgets.text("schema", "forecast")
dbutils.widgets.text("volume", "dais_ts")
dbutils.widgets.text("kaggle_username", "anaprok")
dbutils.widgets.text("kaggle_token", "YOUR_TOKEN")

# COMMAND ----------

import os
os.environ["KAGGLE_USERNAME"] = dbutils.widgets.get("kaggle_username")
os.environ["KAGGLE_KEY"] = dbutils.widgets.get("kaggle_token")

# Defining target catalog and schema
default_catalog =  dbutils.widgets.get("catalog")
default_schema  = dbutils.widgets.get("schema")
default_volume = dbutils.widgets.get("volume")
database_name = f"{default_catalog}.{default_schema}"
volume_path = f"/Volumes/{default_catalog}/{default_schema}/{default_volume}"
MAIN_DIR = volume_path

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Your Data From Kaggle

# COMMAND ----------

# MAGIC %sh 
# MAGIC kaggle datasets files pratyushakar/rossmann-store-sales

# COMMAND ----------

# MAGIC %sh 
# MAGIC kaggle datasets files dromosys/rossmann-store-extra 

# COMMAND ----------


!mkdir {MAIN_DIR}
!kaggle datasets download pratyushakar/rossmann-store-sales -p {MAIN_DIR} --force
!kaggle datasets download dromosys/rossmann-store-extra -p {MAIN_DIR} --force
!ls {MAIN_DIR}
!unzip {MAIN_DIR}/rossmann-store-sales.zip -d {MAIN_DIR}
!unzip {MAIN_DIR}/rossmann-store-extra.zip -d {MAIN_DIR}


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Saving your data under Delta Tables

# COMMAND ----------

from pyspark.sql.types import LongType, StringType, StructField, StructType, ShortType, DateType, IntegerType
from pyspark.sql.functions import year, month, col, lit

customSchema = StructType([
  StructField("Store", StringType()),
  StructField("State", StringType())]
)
store_state_df = spark.read.format("csv")\
  .option("delimiter",",")\
  .option("header", "true")\
  .schema(customSchema)\
  .load(f"{MAIN_DIR}/store_states.csv")

customSchema = StructType([
  StructField("State_Name", StringType()),
  StructField("State", StringType())]
)
state_names_df = spark.read.format("csv")\
  .option("delimiter",",")\
  .option("header", "true")\
  .schema(customSchema)\
  .load(f"{MAIN_DIR}/state_names.csv")

store_state_names_df = store_state_df.join(state_names_df, store_state_df.State == state_names_df.State).select(store_state_df["*"], state_names_df.State_Name)

display(store_state_names_df)

# COMMAND ----------


customSchema = StructType([
  StructField("Store", IntegerType()),
  StructField("StoreType", StringType())]
)
state_df = spark.read.format("csv")\
  .option("delimiter", ",")\
  .option("header", "true")\
  .schema(customSchema)\
  .load(f"{MAIN_DIR}/store.csv")


customSchema = StructType([
  StructField("Store", IntegerType()),
  StructField("DayOfWeek", IntegerType()),
  StructField("Date", DateType()),
  StructField("Sales", IntegerType()),
  StructField("Customers", IntegerType()),
  StructField("Open", ShortType()),
  StructField("Promo", ShortType()),
  StructField("StateHoliday", IntegerType()),
  StructField("SchoolHoliday", IntegerType())]
)
train_df = spark.read.format("csv")\
  .option("delimiter",",")\
  .option("header", "true")\
  .schema(customSchema)\
  .load(f"{MAIN_DIR}/train.csv")

train_df = (train_df
    .withColumn("year", year(col("Date")))\
    .withColumn("month", month(col("Date"))))

# State Holliday is 0 so we are dropping it 
train_df = train_df.join(state_df, train_df.Store == state_df.Store).select(train_df["*"], state_df.StoreType).drop("StateHoliday")
train_df = train_df.join(store_state_names_df, train_df.Store == store_state_names_df.Store).select(train_df["*"], store_state_names_df.State_Name, store_state_names_df.State).withColumn("Store", col("Store").cast("string"))

display(train_df)

# COMMAND ----------

customSchema = StructType([
  StructField("Store", StringType()),
  StructField("StoreType", StringType()),
  StructField("Assortment", StringType()),
  StructField("CompetitionDistance", IntegerType()),
  StructField("CompetitionOpenSinceMonth", IntegerType()),
  StructField("CompetitionOpenSinceYear", IntegerType()),
  StructField("Promo2", IntegerType()),
  StructField("Promo2SinceWeek", IntegerType()),
  StructField("Promo2SinceYear", StringType()),
  StructField("PromoInterval", StringType())]
)
store_df = spark.read.format("csv")\
  .option("delimiter",",")\
  .option("header", "true")\
  .schema(customSchema)\
  .load(f"{MAIN_DIR}/store.csv")

display(store_df)


# COMMAND ----------

customSchema = StructType([
  StructField("StateName", StringType()),
  StructField("Date", DateType()),
  StructField("Max_TemperatureC", IntegerType()),
  StructField("Mean_TemperatureC", IntegerType()),
  StructField("Min_TemperatureC", IntegerType()),
  StructField("Dew_PointC", IntegerType()),
  StructField("MeanDew_PointC", IntegerType()),
  StructField("Min_DewpointC", IntegerType()),
  StructField("Max_Humidity", IntegerType()),
  StructField("Mean_Humidity", IntegerType())
])

weather_df = spark.read.format("csv")\
  .option("delimiter",",")\
  .option("header", "true")\
  .schema(customSchema)\
  .load(f"{MAIN_DIR}/weather.csv")

weather_df = weather_df.join(store_state_names_df, weather_df.StateName == state_names_df.State_Name).drop("StateName")

display(weather_df)

# COMMAND ----------

writing_tables = None
if writing_tables:
  train_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{default_catalog}.{default_schema}.main_dataset_stores")
  weather_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{default_catalog}.{default_schema}.extra_dataset_weather")
  store_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{default_catalog}.{default_schema}.extra_dataset_stores")


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Creating new features with underlying Delta Tables
# MAGIC

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {database_name}.main_sales_fs")
spark.sql(f"DROP TABLE IF EXISTS {database_name}.main_features_sales_fs")
spark.sql(f"DROP TABLE IF EXISTS {database_name}.extra_sales_weather_fs")

# COMMAND ----------

main_sales = spark.read.table(f"{database_name}.main_dataset_stores").select("Store", "Date", "Sales")
main_sales_features = spark.read.table(f"{database_name}.main_dataset_stores").drop("Sales")
main_weather_features = spark.read.table(f"{database_name}.extra_dataset_weather").select( "Store", "Date", "State", "Mean_TemperatureC", "MeanDew_PointC", "Mean_Humidity")


# COMMAND ----------

# Creating Feature Stores per 2 keys - Store, Date 
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient(model_registry_uri="databricks-uc")

fe.create_table(
    name=f"{default_catalog}.{default_schema}.main_sales_fs",
    primary_keys=["Store", "Date"],
    df=main_sales,
    description="Main sales stores dataset with the timestamp",
    tags={"team":"sales"}
)

fe.create_table(
    name=f"{default_catalog}.{default_schema}.main_features_sales_fs",
    primary_keys=["Store", "Date"],
    df=main_sales_features,
    description="Main features for sales stores dataset with the timestamp",
    tags={"team":"sales"}
)

fe.create_table(
    name=f"{default_catalog}.{default_schema}.extra_sales_weather_fs",
    primary_keys=["Store", "Date"],
    df=main_weather_features,
    description="Weather forecast to the stores locations with the timestamp",
    tags={"team":"sales"}
)


# COMMAND ----------


