#****************************************************************************
# (C) Cloudera, Inc. 2020-2022
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

from __future__ import print_function
import time
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint
import os
import json

client = cmlapi.default_client()

client.list_projects()

project_id = os.environ['CDSW_PROJECT_ID']

# List Registered Models at Workspace Level

try:
    # List registered models.
    api_response = client.list_registered_models()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_registered_models: %s\n" % e)

    
# List Registered Models at Workspace Level filter by username:

try:
    # List registered models.
    search_filter = {"creator_id":"vishrajagopalan"}
    search = json.dumps(search_filter)
    page_size = 100
    api_response = client.list_registered_models(search_filter=search, page_size=page_size)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_registered_models: %s\n" % e)
    
# List ALL Experiments

try:
    # Lists all experiments that belong to a user across all projects.
    search_filter = {"name":"xgboost-classifier"}
    search = json.dumps(search_filter)
    api_response = client.list_all_experiments(search_filter=search)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_all_experiments: %s\n" % e)

try:
    # Lists all experiments that belong to a user across all projects.
    search_filter = {"name":"heart-condition-classifier"}
    search = json.dumps(search_filter)
    api_response = client.list_all_experiments(search_filter=search)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_all_experiments: %s\n" % e)
    

# List Projects    

try:
    # Return all projects, optionally filtered, sorted, and paginated.
    search_filter = {"owner.name" : "Paul de Fusco"}
    search = json.dumps(search_filter)
    api_response = client.list_projects(search_filter=search)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_projects: %s\n" % e)

# List Experiments
    
try:
    # List all experiments in a given project.
    api_response = client.list_experiments(project_id = os.environ["CDSW_PROJECT_ID"])
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_experiments: %s\n" % e)

    
experiment_id = "mwzb-4nfp-me2y-h3z0" # str | Experiment ID to search over.
#search_filter = 'search_filter_example' # str | Search filter is an optional HTTP parameter to filter results by. Supported search filter keys are: [creator.email creator.name creator.username name status]. Dynamic search key words are supported for experiment runs. Supported fields are [metrics tags params]. (optional)
page_size = 40 # int | Page size is an optional argument for number of entries to return in one page. If not specified, the server will determine a page size. If specified, must be respecified for further requests when using the provided next page token in the response. (optional)

try:
    # Returns a list of Runs that belong to an experiment.
    api_response = client.list_experiment_runs(project_id, experiment_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_experiment_runs: %s\n" % e)

    
## Register a Model from an Experiment
## This provides an Entry in the MLFlow Model Registry
  
CreateRegisteredModelRequest = {
                                "project_id": project_id, 
                                "experiment_id" : experiment_id,
                                "run_id": "sccl-5z6v-2tin-wtn1", 
                                "model_name": "wine_model_from_api", 
                                "model_path": "artifacts"
                               }

try:
    # Register a model.
    api_response = client.create_registered_model(CreateRegisteredModelRequest)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->create_registered_model: %s\n" % e)
    
    
## LIST REGISTERED MODELS

sort = 'created_at' # str | Sort is an optional HTTP parameter to sort results by. Supported sort keys are: [created_at creator.email creator.name creator.username description kernel name script status updated_at]. where \"+\" means sort by ascending order, and \"-\" means sort by descending order. For example:   sort=-created_at. (optional)

try:
    # List registered models.
    api_response = client.list_registered_models(sort=sort)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_registered_models: %s\n" % e)

    
## Create a Model from the Registry
    
#project_id	str	ID of the project containing the model.	[optional]
#name	str	Name of the model.	[optional]
#description	str	Description of the model.	[optional]
#disable_authentication	bool	Whether to disable authentication for requests to deployments of this model.	[optional]
#registered_model_id	str	Registered Model ID.
  
CreateModelRequest = {
                        "project_id": project_id, 
                        "name" : "wine_model_from_api",
                        "description": "my wine clf from api", 
                        "registered_model_id": "4gww-khqa-ulsf-7x2x"
                     }

try:
    # Create a model.
    api_response = client.create_model(CreateModelRequest, project_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->create_model: %s\n" % e)

    
# create an instance of the API class
sort = 'created_at' # str | Sort is an optional HTTP parameter to sort results by. Supported sort keys are: [auth_enabled created_at creator.email creator.name creator.username description name updated_at]. where \"+\" means sort by ascending order, and \"-\" means sort by descending order. For example:   sort=created_at. (optional)
page_size = 56 # int | page size of the response model build list. (optional)

try:
    # List models, optionally filtered, sorted, and paginated.
    api_response = client.list_models(project_id, sort=sort, page_size=page_size)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_models: %s\n" % e)

    
try:
    # List the available runtimes, optionally filtered, sorted, and paginated.
    api_response = client.list_runtimes()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_runtimes: %s\n" % e)

    
try:
    # List the available runtime addons, optionally filtered, sorted, and paginated.
    api_response = client.list_runtime_addons()
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_runtime_addons: %s\n" % e)
    
    
model_id = "73092171-1b50-4317-95fe-983937c6ab90"

CreateModelBuildRequest = {
                            "registered_model_version_id": "4gww-khqa-ulsf-7x2x", 
                            "runtime_addon_identifiers": ["hadoop-cli-7.2.14-hf9"], 
                            "runtime_identifier": "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.9-standard:2023.08.2-b8",
                            "comment": "invoking model build"
                          }

try:
    # Create a model build.
    api_response = client.create_model_build(CreateModelBuildRequest, project_id, model_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->create_model_build: %s\n" % e)

    
model_id = "73092171-1b50-4317-95fe-983937c6ab90" # str | ID of the model to get builds for.
sort = 'created_at' # str | Sort is an optional HTTP parameter to sort results by. Supported sort keys are: [built_at comment created_at creator.email creator.name creator.username crn status updated_at]. where \"+\" means sort by ascending order, and \"-\" means sort by descending order. For example:   sort=creator.email. (optional)

try:
    # List model builds, optionally filtered, sorted, and paginated.
    api_response = client.list_model_builds(project_id, model_id, sort=sort)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->list_model_builds: %s\n" % e)


# create an instance of the API class
model_id = "73092171-1b50-4317-95fe-983937c6ab90" # str | ID of the model to deploy.
build_id = "1c1818d4-91ab-4d19-9659-e03846c630fe" # str | ID of the model build to deploy.

CreateModelDeploymentRequest = {
  "cpu":"2",
  "memory":"4"
}

try:
    # Create a model deployment.
    api_response = client.create_model_deployment(CreateModelDeploymentRequest, project_id, model_id, build_id)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling CMLServiceApi->create_model_deployment: %s\n" % e)

    
