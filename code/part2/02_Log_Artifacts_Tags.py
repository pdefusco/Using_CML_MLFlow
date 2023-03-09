import json
import os
import mlflow

# Create some files to preserve as artifacts
features = "rooms, zipcode, median_price, school_rating, transport"
data = {"state": "TX", "Available": 25, "Type": "Detached"}

# Create couple of artifact files under the directory "data"
os.makedirs("data", exist_ok=True)
with open("data/data.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
with open("data/features.txt", 'w') as f:
    f.write(features)

tags = {
    "engineering": "ML Platform",
    "release.candidate": "RC1",
    "release.version": "2.2.0",
}

with mlflow.start_run():
  # Set a batch of tags
  mlflow.set_tags(tags)    
    
  # Write all files in "data" to root artifact_uri/states
  mlflow.log_artifacts("data", artifact_path="states")