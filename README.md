# ML Flow Experiments in CML

This project is a CML Quickstart for ML Flow Experiments based on the Release Guide that can be downloaded at this [link](https://docs.cloudera.com/cdp-public-cloud-preview-features/cloud/pub-ml-experiments-with-mlflow/pub-ml-experiments-with-mlflow.pdf).

## Overview

Machine Learning requires experimenting with a wide range of datasets, data preparation steps,
and algorithms to build a model that maximizes a target metric. Once you have built a model,
you also need to deploy it to a production system, monitor its performance, and continuously
retrain it on new data and compare it with alternative models.

CML lets you train, reuse, and deploy models with any library, and package them into
reproducible artifacts that other data scientists can use.
CML packages the ML models in a reusable, reproducible form so you can share it with other
data scientists or transfer it to production.

CML is compatible with the MLflow tracking API and makes use of the MLflow client library as
the default method to log experiments. Existing projects with existing experiments are still
available and usable.

The functionality described in this document is for the new version of the Experiments feature,
which replaces an older version of the Experiments feature that could not be used from within
Sessions. In Projects that have existing Experiments created using the previous feature, you
can continue to view these existing Experiments. New Projects will use the new Experiments
feature.

## CML Experiment Tracking through MLflow API

Note: CML currently supports only Python for experiment tracking.

CML’s experiment tracking features allow you to use the MLflow client library for logging
parameters, code versions, metrics, and output files when running your machine learning code.
The MLflow library is available in CML Sessions without you having to install it. CML also
provides a UI for later visualizing the results. MLflow tracking lets you log and query
experiments using the following logging functions:

● mlflow.create_experiment()creates a new experiment and returns its ID. Runs
can be launched under the experiment by passing the experiment ID to
mlflow.start_run.
Cloudera recommends that you create an experiment to organize your runs. You can
also create experiments using the UI.

● mlflow.set_experiment() sets an experiment as active. If the experiment does not
exist, mlflow.set_experiment creates a new experiment. If you do not wish to use
the set_experiment method, default experiment is selected.
Cloudera recommends that you set the experiment using mlflow.set_experiment.

● mlflow.start_run() returns the currently active run (if one exists), or starts a new
run and returns a mlflow.ActiveRun object usable as a context manager for the
current run. You do not need to call start_run explicitly: calling one of the logging
functions with no active run automatically starts a new one.

● mlflow.end_run() ends the currently active run, if any, taking an optional run status.

● mlflow.active_run() returns a mlflow.entities.Run object corresponding to
the currently active run, if any.
Note: You cannot access currently-active run attributes (parameters, metrics, etc.)
through the run returned by mlflow.active_run. In order to access such attributes,
use the mlflow.tracking.MlflowClient as follows:
client = mlflow.tracking.MlflowClient()
data = client.get_run(mlflow.active_run().info.run_id).data

● mlflow.log_param() logs a single key-value parameter in the currently active run.
The key and value are both strings. Use mlflow.log_params() to log multiple
parameters at once.

● mlflow.log_metric() logs a single key-value metric for the current run. The value
must always be a number. MLflow remembers the history of values for each metric. Use
mlflow.log_metrics()to log multiple metrics at once.
Parameters:
○ key - Metric name (string)
○ value - Metric value (float). Note that some special values such as +/- Infinity
may be replaced by other values depending on the store. For example, the
SQLAlchemy store replaces +/- Infinity with max / min float values.
○ step - Metric step (int). Defaults to zero if unspecified.
Syntax - mlflow.log_metrics(metrics: Dict[str, float], step:
Optional[int] = None) → None

● mlflow.set_tag() sets a single key-value tag in the currently active run. The key
and value are both strings. Use mlflow.set_tags() to set multiple tags at once.

● mlflow.log_artifact() logs a local file or directory as an artifact, optionally taking
an artifact_path to place it within the run’s artifact URI. Run artifacts can be organized
into directories, so you can place the artifact in a directory this way.

● mlflow.log_artifacts() logs all the files in a given directory as artifacts, again
taking an optional artifact_path.

● mlflow.get_artifact_uri() returns the URI that artifacts from the current run
should be logged to.
For more information on MLflow API commands used for tracking, see MLflow Tracking
