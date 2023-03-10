import mlflow

client = mlflow.tracking.MlflowClient()
#data = client.get_run(mlflow.latest_active_run().info.run_id).data

experiment_id = "w2q8-1vw8-lruw-nx26"
experiment = client.get_experiment(experiment_id)

print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
