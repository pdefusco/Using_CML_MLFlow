!pip3 install -r requirements.txt

import os
import warnings
import sys
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.spark
import logging
import json
import shutil
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

#mlflow.end_run()

if __name__ == "__main__":

  spark = SparkSession.builder.config("spark.jars.packages", "org.mlflow:mlflow-spark:2.2.1")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-2")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://go01-demo")\
    .getOrCreate()
  
  mlflow.set_experiment("sparkml-experiment")

  training = spark.createDataFrame(
      [
          (0, "a b c d e spark", 1.0),
          (1, "b d", 0.0),
          (2, "spark f g h", 1.0),
          (3, "hadoop mapreduce", 0.0),
      ],
      ["id", "text", "label"],
  )
  
  with mlflow.start_run() as run:
  
    maxIter=10
    regParam=0.001
  
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=maxIter, regParam=regParam)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    model = pipeline.fit(training)

    mlflow.log_param("maxIter", maxIter)
    mlflow.log_param("regParam", regParam)

    #prediction = model.transform(test)
    
    mlflow.spark.log_model(model, "iceberg-sparkml-model", registered_model_name="IcebergSparkMLModel")

    
#spark.stop()