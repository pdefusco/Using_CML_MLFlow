import mlflow
import pandas as pd

logged_model = '/home/cdsw/.experiments/0trj-c5b3-3bni-kqbv/wgr0-r0cb-24lk-8kf2/artifacts/model'

def predict(args):

  # Load model as a PyFuncModel.
  data = args.get('input')
  loaded_model = mlflow.pyfunc.load_model(logged_model)
  # Predict on a Pandas DataFrame.
  return loaded_model.predict(pd.DataFrame(data))
