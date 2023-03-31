import mlflow

logged_model = '/home/cdsw/.experiments/0trj-c5b3-3bni-kqbv/wgr0-r0cb-24lk-8kf2/artifacts/model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

data_score = data.drop(columns=['quality'])

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data_score))