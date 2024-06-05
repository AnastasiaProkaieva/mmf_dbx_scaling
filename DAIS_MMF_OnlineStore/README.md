# Project Description 



## Main Problems 



## Solutions 



> **Chain of thoughts:** The code follows the structure below:
>- create a `pyfunc` class that allows calling the models into one endpoint. this class managed the 100K's of models.
>- store the models in the `sales_models_fs` (this table is also stored as artifact on the model registry for reference).
>- register a Feature Spec table for the model table so that the models can be called from the predict when required (offline or online).
>- retrain when required the models and update the model table.
>- sync the offline delta table of the models to the online table.
>- the individual models **are not** recorded in MLFlow to reduce overhead. this is OK for 10's of models but not in this case where we have 100K's of models.

**Keep in mind** that if you want to do only `fit().predict()` at each evocation you need to serialize only 1 model where you include fit under the `.predict` and you dont have to serialize dictionary of artifacts since you do not require an object per model (it will be created at each fit during the serving). This approach does not allow to have a traceability of the model and reusing it between offline and online forecasting


The code has been done to follow the following pattern: evoke 1 object with all pretrained models that is based on the ID of interest that contains independently pretrained forecasting models.


Step per step approach: 
- you train your models independently and store them under Experiment of MlFlow with their corresponding RunIDs
  - each experiment correspond to a FIT of your model 
    - we would initiate a new experiment each time we are recreating a new model(using FIT)
  - each run ID = 1 model




**Keep in mind** that if you want to do only `fit().predict()` at each evokation you need to serialize only 1 model where you include fit under the `.predict` and you dont have to serialize dictionary of artifacts since you do not require an object per model (it will be created at each fit during the serving). <br>
There are a few things to keep in mind with that approach: 