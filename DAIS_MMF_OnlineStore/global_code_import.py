# Databricks notebook source
dbutils.widgets.dropdown("reset_tables", "False", ["True", "False"], label="Set reset status for tables")
dbutils.widgets.text("catalog", "ap")
dbutils.widgets.text("schema", "forecast")

default_catalog =  dbutils.widgets.get("catalog")
default_schema  = dbutils.widgets.get("schema")
database_name = f"{default_catalog}.{default_schema}"

# COMMAND ----------

print("Loading Feature Tables")

# COMMAND ----------


# Creating the training dataset using the feature lookup - this definition is later recorded on the model artifact at training time and serves as the ground for later inference and reconciliation between offline and online serving

# Grabbing the dataset for training

fe = FeatureEngineeringClient()

label_df = (spark
               .table(f"{database_name}.main_sales_fs")
               .select("Store","Date","Sales"))

feature_lookups = [ 
  FeatureLookup(
      table_name=f"{database_name}.main_features_sales_fs", 
      lookup_key=["Store", "Date"],
      feature_names=["SchoolHoliday", "Promo"]
  ),
  FeatureLookup(
      table_name=f"{database_name}.extra_sales_weather_fs",  
      lookup_key=["Store", "Date"],
      feature_names=["Mean_TemperatureC"]
  ), 
]

training_set = fe.create_training_set(
    df=label_df,
    feature_lookups=feature_lookups,
    label='Sales',
)
training_df = training_set.load_df()


models_lookups = [FeatureLookup(
      table_name=f"{database_name}.sales_model_table_main",  
      lookup_key=["Store"],
      feature_names=["encoded_model"]
  ),
]

features_set = fe.create_training_set(
    df=label_df,
    feature_lookups=feature_lookups+models_lookups,
    label='Sales',
)
features_df = features_set.load_df()


# COMMAND ----------

print("Pre-creating mock tests")

# COMMAND ----------

# example for 1 model to register the input example
store_id = '1001'

data_mocking_test = training_set.load_df().filter(f"Store == {store_id}").toPandas()

# Get list of model from model table
model_list = spark.table(f"{database_name}.sales_model_table_main").toPandas() # for using the latest version

# artifact needs to be recorded as a JSON in model registry -- transforming model pandas dataframe into dict
model_list['training_date'] = [date.strftime('%Y-%m-%d') for date in model_list['training_date']]
model_list_dict = model_list.to_dict(orient='records')
model_list = model_list.set_index('Store') # setting the index for using later below 

# generating an example to be stored in the model registry
model = pickle.loads(urlsafe_b64decode(model_list.loc[store_id, 'encoded_model'][0].encode("utf-8"))) # [0]
predict_df = model.predict(context=None, data_input = data_mocking_test)

# Generate signature
data_mocking_test['Date'] = data_mocking_test['Date'].astype('datetime64[ns]')
data_mocking_test['Store'] = data_mocking_test['Store'].astype('string')

signature = infer_signature(
  data_mocking_test[['Date', 'Store']].iloc[0:10],
  pd.concat([data_mocking_test[['Date', 'Store']].iloc[0:10], pd.DataFrame([pd.Series(range(10))], columns=['Sales_Pred'])])
)

# COMMAND ----------

# def create_online_table(table_name, pks, timeseries_key=None, delete_old=None, spec_name="sales_models_table_feature_spec"):
  
#     w = WorkspaceClient(token=db_token, host=db_host)
#     online_table_name = table_name+"_online"

#     if delete_old:
#       fe.delete_feature_spec(name=f"{database_name}.{spec_name}")
#     from databricks.sdk.service import catalog as c
#     print(f"Creating online table for {online_table_name}...")
#     try:
#       spark.sql(f'ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')
#     except:
#       print("Table is already CDF enabled")
#     try:
#       spec = c.OnlineTableSpec(
#           source_table_full_name=table_name, 
#           primary_key_columns=pks, 
#           run_triggered={'triggered': 'true'}, 
#           timeseries_key=timeseries_key
#           )
#       w.online_tables.create(name=online_table_name, spec=spec)
#     except:
#       print(f"Seems like your Online table {online_table_name} already exist")
