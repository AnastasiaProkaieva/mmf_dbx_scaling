## Description 
This is a supportive material for the Blog Post [Serve Many Forecasting Models with Databricks Model Serving at Once](https://medium.com/p/80d5e3f32943/edit).


This code walks you through multiple appoaches such as : 
- How to train many independent forecasting models on top of Delta table using PandasUDF of Apache Spark.
- How to track each model using MLFlow and Delta tables. 
- How to serialize all your models under a single MLFlow object. 
- How to prepare your input data payload to be consumed by a single Endpoint.

The dataset used for this code can be found [here](https://www.kaggle.com/c/rossmann-store-sales) on the main Kaggle page. 
