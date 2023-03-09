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
import datetime
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

#mlflow.end_run()

if __name__ == "__main__":

  spark = SparkSession.builder.appName("Iceberg-Spark-MLFlow")\
    .config("spark.jars.packages", "org.mlflow:mlflow-spark:2.2.1")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-2")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://go01-demo")\
    .config("spark.sql.extensions","org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog","org.apache.iceberg.spark.SparkSessionCatalog") \
    .config("spark.sql.catalog.local","org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type","hadoop") \
    .config("spark.sql.catalog.spark_catalog.type","hive") \
    .getOrCreate()

  mlflow.set_experiment("sparkml-experiment")

  training_df = spark.createDataFrame(
      [
          (0, "a b c d e spark", 1.0),
          (1, "b d", 0.0),
          (2, "spark f g h", 1.0),
          (3, "hadoop mapreduce", 0.0),
      ],
      ["id", "text", "label"],
  )

  training_df.writeTo("spark_catalog.default.training_data").using("iceberg").createOrReplace()
  spark.sql("SELECT * FROM spark_catalog.default.training_data").show()

  ### SHOW TABLE HISTORY AND SNAPSHOTS
  spark.read.format("iceberg").load("spark_catalog.default.training_data.history").show(20, False)
  spark.read.format("iceberg").load("spark_catalog.default.training_data.snapshots").show(20, False)

  ### ICEBERG INSERT DATA - APPEND FROM DATAFRAME
  #temp_df = spark.sql("SELECT * FROM spark_catalog.default.training_data").sample(fraction=0.3, seed=3)
  #temp_df.writeTo("spark_catalog.default.training_data").append()

  #spark.read.format("iceberg").load("spark_catalog.default.training_data.history").show(20, False)
  #spark.read.format("iceberg").load("spark_catalog.default.training_data.snapshots").show(20, False)

  snapshot_id = spark.read.format("iceberg").load("spark_catalog.default.training_data.snapshots").select("snapshot_id").tail(1)[0][0]
  committed_at = spark.read.format("iceberg").load("spark_catalog.default.home_credit_credit_card_balance.snapshots").select("committed_at").tail(1)[0][0]
  parent_id = spark.read.format("iceberg").load("spark_catalog.default.home_credit_credit_card_balance.snapshots").select("parent_id").tail(1)[0][0]

  tags = {
      "iceberg_snapshot_id": snapshot_id,
      "iceberg_snapshot_committed_at": committed_at.strftime('%m/%d/%Y'),
      "iceberg_parent_id": parent_id,
  }

  ### MLFLOW EXPERIMENT RUN
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
    mlflow.set_tags(tags)
    mlflow.spark.log_model(model, "iceberg-sparkml-model", registered_model_name="IcebergSparkMLModel")

  mlflow.end_run()




#spark.stop()
