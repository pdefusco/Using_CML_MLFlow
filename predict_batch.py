import mlflow

logged_model = '/home/cdsw/.experiments/umyb-mq5h-2v3x-j445/5t4w-q921-u7rz-1jfz/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

data_score = data.drop(columns=['quality'])

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data_score))