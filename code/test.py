import mlflow

mlflow.sklearn.log_model(lr, "model", registered_model_name="Wine Quality Model")