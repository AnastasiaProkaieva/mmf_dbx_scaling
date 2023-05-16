# Databricks notebook source
# MAGIC %pip install kaggle

# COMMAND ----------

import os
os.environ["KAGGLE_USERNAME"] = "PLEASE ENTER YOUR USER NAME"
os.environ["KAGGLE_KEY"] = "PLEASE ENTER YOUR SECRET TOKEN"

# COMMAND ----------

# MAGIC %sh 
# MAGIC kaggle competitions files rossmann-store-sales

# COMMAND ----------

MAIN_DIR = "/dbfs/FileStore/Users/anastasia.prokaieva@databricks.com/serving_mmf_dataset"

# COMMAND ----------

!mkdir {MAIN_DIR}
!kaggle competitions download rossmann-store-sales -p {MAIN_DIR} --force
!ls {MAIN_DIR}
!unzip {MAIN_DIR}/rossmann-store-sales.zip -d {MAIN_DIR}

# COMMAND ----------

!head {MAIN_DIR}/train.csv

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS ap;
# MAGIC DROP TABLE IF EXISTS ap.rossmann_train;
# MAGIC DROP TABLE IF EXISTS ap.rossmann_test;

# COMMAND ----------


from pyspark.sql.types import LongType, StringType, StructField, StructType, ShortType, DateType, IntegerType
from pyspark.sql.functions import year, month, col

customSchema = StructType([
  StructField("Store", StringType()),
  StructField("State", StringType())]
)
state_df = spark.read.format("csv")\
  .option("delimiter", ",")\
  .option("header", "true")\
  .schema(customSchema)\
  .load(f"/FileStore/Users/anastasia.prokaieva@databricks.com/serving_mmf_dataset/store.csv")


# COMMAND ----------

customSchema = StructType([
  StructField("Store", StringType()),
  StructField("DayOfWeek", IntegerType()),
  StructField("Date", DateType()),
  StructField("Sales", IntegerType()),
  StructField("Customers", IntegerType()),
  StructField("Open", ShortType()),
  StructField("Promo", ShortType()),
  StructField("StateHoliday", StringType()),
  StructField("SchoolHoliday", ShortType())]
)
train_df = spark.read.format("csv")\
  .option("delimiter",",")\
  .option("header", "true")\
  .schema(customSchema)\
  .load(f"/FileStore/Users/anastasia.prokaieva@databricks.com/serving_mmf_dataset/train.csv")

train_df = (train_df
    .withColumn("year", year(col("Date")))\
    .withColumn("month", month(col("Date"))))

train_df = train_df.join(state_df, train_df.Store == state_df.Store).select(train_df["*"], state_df.State)


write2storage = True
if write2storage:
    train_df.write.mode("overwrite").saveAsTable("hive_metastore.ap.hack_ap_rossmann_blog")

# COMMAND ----------

display(train_df)

# COMMAND ----------


