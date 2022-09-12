
import logging
from collections import namedtuple
from pathlib import Path
from kfp.v2.dsl import Output, Metrics, ClassificationMetrics
from query_data import query_data
from train_model import train_model
from evaluation import evaluation
from unittest.mock import Mock


def run_local(project_id):
    
    output_parquet_train_set = Mock(path='gs://test_iris_data/train_set.pq')
    output_parquet_test_set = Mock(path='gs://test_iris_data/test_set.pq')
    
    query_data.python_func(project_id=project_id,
                           output_parquet_train_set=output_parquet_train_set,
                           output_parquet_test_set=output_parquet_test_set)

    input_parquet_train_set = Mock(path='gs://test_iris_data/train_set.pq')
    output_model= Mock(path='C:/repos/tmp//model.pickle')
    train_model.python_func(project_id=project_id,input_parquet_train_set=input_parquet_train_set,output_model=output_model)
    
    input_model= Mock(path='C:/repos/tmp/model.pickle')
    #input_model= Mock(path='C:/repos/tmp/model.pickle')
    input_test_set = Mock(path='gs://test_iris_data/test_set.pq')
    output_metrics = Mock(path='gs://test_iris_data/outputMatrics')
    output_pic=Mock(path='gs://test_iris_data/pictures')
    markdown_artifact=Mock(path='gs://test_iris_data/markdown')
    evaluation.python_func(input_model=input_model,
                   input_test_set=input_test_set,
                   output_metrics=output_metrics,
                   output_pic=output_pic,
                   markdown_artifact=markdown_artifact)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    run_local(project_id='ranathakur123')
