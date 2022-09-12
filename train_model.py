from datetime import datetime

# imports For building components
from kfp.v2.dsl import component
# Type annotations for the component artifacts
from kfp.v2.dsl import (
    Input,
    Output,
    Artifact,
    Dataset,
    Model,
    Metrics,
    Markdown,
    OutputPath
)
import os
credential_path = "C:\\repos\\analysis\\test-iris-classification-example.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

@component(
    output_component_file="train_data.yaml",
    base_image="gcr.io/ranathakur123/test1_ml_image_kfp1/0220901"
)
def train_model(project_id: str,
               input_parquet_train_set:Input[Dataset],
               output_model: Output[Model]):

    import pandas as pd
    from google.cloud import bigquery
    from lightgbm import LGBMClassifier, LGBMRegressor
    import pickle
    import logging
    
    logging.info("train_model component run started.")
    print('1.Loading data')
    train_set = pd.read_parquet(input_parquet_train_set.path)
    X_train=train_set.iloc[:,0:4]
    y_train=train_set.iloc[:,-1]
    
    #Creating a lightgbm model and training
    print('2.training the model')
    model=LGBMClassifier()
    model.fit(X_train, y_train)
    
    print('3.Saving the model')
    pickle.dump(model, open(output_model.path, 'wb'))
    
    logging.info("train_model component run complete.")