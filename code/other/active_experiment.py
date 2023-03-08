import mlflow

mlflow.start_run()
run = mlflow.active_run()
print("Active run_id: {}".format(run.info.run_id))
mlflow.end_run()