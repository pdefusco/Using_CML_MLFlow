import mlflow
import pandas as pd

logged_model = '/home/cdsw/.experiments/umyb-mq5h-2v3x-j445/5t4w-q921-u7rz-1jfz/artifacts/model'

def predict(args):
  
  # Load model as a PyFuncModel.
  data = args.get('input')
  loaded_model = mlflow.pyfunc.load_model(logged_model)
  # Predict on a Pandas DataFrame.
  return loaded_model.predict(pd.DataFrame(data))