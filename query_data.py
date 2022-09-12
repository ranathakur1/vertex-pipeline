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
#credential_path = "C:\\repos\\analysis\\mtech-algo-retail-poc-c51d3b748e05.json"
credential_path = "C:\\repos\\analysis\\test-iris-classification-example.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

@component(
    #output_component_file="train_data.yaml",
    base_image="gcr.io/ranathakur123/test1_ml_image_kfp1/0220901"
)
def query_data(project_id: str,
               output_parquet_train_set:Output[Dataset],
               output_parquet_test_set:Output[Dataset]):
    
    import pandas as pd
    from google.cloud import bigquery
    import logging
    logging.info("query_data component run started.")
    
    print('1.Query data')
    query = """
    select * from bigquery-public-data.ml_datasets.iris
    """
    logging.info("Data query successful")
    client = bigquery.Client(project=project_id)
    transit_set = client.query(query).result().to_dataframe()
    thrshld=int(len(transit_set)*0.7)

    train_set=transit_set.iloc[0:thrshld,:]
    test_set=transit_set.iloc[thrshld:len(transit_set),:]
    
 
    print('2.Writing data')
    train_set.to_parquet(output_parquet_train_set.path, index=False)
    test_set.to_parquet(output_parquet_test_set.path, index=False)
    
    logging.info("query_data component run complete.")

    
