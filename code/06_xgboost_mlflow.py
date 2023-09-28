import os
import warnings
import sys
import xgboost
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow.sklearn
import logging


mlflow.set_experiment(experiment_name="heart-condition-classifier")

file_url = "https://azuremlexampledata.blob.core.windows.net/data/heart-disease-uci/data/heart.csv"
df = pd.read_csv(file_url)
                 
df["thal"] = df["thal"].astype("category").cat.codes
                 
df["thal"].unique()

                 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1), df["target"], test_size=0.3
)
                 
from xgboost import XGBClassifier

with mlflow.start_run():

  model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
  
  model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

  y_pred = model.predict(X_test)

  from sklearn.metrics import accuracy_score, recall_score

  accuracy = accuracy_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)                 

  print("Accuracy: %.2f%%" % (accuracy * 100.0))
  print("Recall: %.2f%%" % (recall * 100.0))

  
  mlflow.log_param("accuracy", accuracy)
  mlflow.log_param("recall", recall)
  mlflow.xgboost.log_model(model, artifact_path="artifacts", registered_model_name="my_xgboost_model")
  

                 
run = mlflow.get_run("n5rt-lafm-0dkn-uw2t")

pd.DataFrame(data=[run.data.params], index=["Value"]).T
                 
                 
pd.DataFrame(data=[run.data.metrics], index=["Value"]).T
                 
client = mlflow.tracking.MlflowClient()
client.list_artifacts(run_id=run.info.run_id)

file_path = mlflow.artifacts.download_artifacts(
    run_id=run.info.run_id, artifact_path="feature_importance_weight.png"
)
                 
                 
import matplotlib.pyplot as plt
import matplotlib.image as img

image = img.imread(file_path)
plt.imshow(image)
plt.show()