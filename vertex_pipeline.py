
from datetime import datetime
import logging
from collections import namedtuple
from pathlib import Path
# For building components
from kfp.v2.dsl import component
import google.cloud.aiplatform as aip
from kfp.v2.dsl import Output, Metrics, ClassificationMetrics
# For creating the pipeline
from kfp.v2.dsl import pipeline, Condition

# For creating the pipeline
from kfp.v2 import dsl

from query_data import query_data
from train_model import train_model
from evaluation import evaluation

import google.cloud.aiplatform as aip



@dsl.pipeline(
    name="test-iris-classofication",
)
def my_pipeline(project_id: str):

    query_bq_data_task = query_data(project_id=project_id)
    
    training_task = train_model(project_id=project_id,
                                input_parquet_train_set=query_bq_data_task.outputs['output_parquet_train_set'])
    
    training_task.set_memory_limit("8G")
    training_task.add_node_selector_constraint('cloud.google.com/gke-accelerator', 
                                                'nvidia-tesla-k80')
    training_task.set_gpu_limit(1)
    
    evaluation_task = evaluation(input_model=training_task.outputs['output_model'],
                                 input_test_set=query_bq_data_task.outputs['output_parquet_test_set'])
    
    
if __name__ == '__main__':
    # to execute the same code using vertex ai use v2 compiler, vertex will create a temporary GKE and manage it. Once the pipeline will finish, it will delete this cluster.
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=my_pipeline,
        package_path="gs://test_iris_classification/test-iris-classification.json",
    )
    
    TIMESTAMP = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    BUCKET_NAME = 'gs://test_iris_classification'
    PROJECT_ID = 'ranathakur123'
    # under the pipeline root all artifacts will be stored
    PIPELINE_ROOT = f"{BUCKET_NAME}/del-artifact-pipeline"
    
    aip.init(project=PROJECT_ID, staging_bucket=BUCKET_NAME)
    
    job = aip.PipelineJob(
        display_name='test-iris-classification',
        template_path="test-iris-classification.json",
        job_id=f"test-iris-classification-{TIMESTAMP}",
        parameter_values={'project_id': PROJECT_ID},
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False)
    
    
    job.run()